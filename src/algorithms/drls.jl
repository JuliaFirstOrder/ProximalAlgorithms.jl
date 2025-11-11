# Themelis, Stella, Patrinos, "Douglas-Rachford splitting and ADMM for
# nonconvex optimization: Accelerated and Newton-type linesearch algorithms",
# Computational Optimization and Applications, vol. 82, no. 2, pp. 395-440 (2022).

function drls_default_gamma(f::Tf, mf, Lf, alpha, lambda) where {Tf}
    if mf !== nothing && mf > 0
        return 1 / (alpha * mf)
    end
    return ProximalCore.is_convex(Tf) ? alpha / Lf : alpha * (2 - lambda) / (2 * Lf)
end

function drls_C(f::Tf, mf, Lf, gamma, lambda) where {Tf}
    a = mf === nothing || mf <= 0 ? gamma * Lf : 1 / (gamma * mf)
    m = ProximalCore.is_convex(Tf) ? max(a - lambda / 2, 0) : 1
    return (lambda / ((1 + a)^2) * ((2 - lambda) / 2 - a * m))
end

"""
    DRLSIteration(; <keyword-arguments>)

Iterator implementing the Douglas-Rachford line-search algorithm [1].

This iterator solves optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

See also: [`DRLS`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `mf=nothing`: convexity modulus of f.
- `Lf=nothing`: Lipschitz constant of the gradient of f.
- `gamma`: stepsize to use, chosen appropriately based on Lf and mf by defaults.
- `max_backtracks=20`: maximum number of line-search backtracks.
- `directions=LBFGS(5)`: strategy to use to compute line-search directions.

# References
1. Themelis, Stella, Patrinos, "Douglas-Rachford splitting and ADMM for nonconvex optimization: Accelerated and Newton-type linesearch algorithms", Computational Optimization and Applications, vol. 82, no. 2, pp. 395-440 (2022).
"""
Base.@kwdef struct DRLSIteration{R,Tx,Tf,Tg,Tmf,TLf,D}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    lambda::R = real(eltype(x0))(1)
    mf::Tmf = nothing
    Lf::TLf = nothing
    gamma::R = drls_default_gamma(f, mf, Lf, alpha, lambda)
    c::R = beta * drls_C(f, mf, Lf, gamma, lambda)
    dre_sign::Int = mf === nothing || mf <= 0 ? 1 : -1
    max_backtracks::Int = 20
    directions::D = LBFGS(5)
end

Base.IteratorSize(::Type{<:DRLSIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct DRLSState{R,Tx,TH}
    x::Tx
    u::Tx
    v::Tx
    w::Tx
    res::Tx
    res_prev::Tx = similar(x)
    xbar::Tx
    xbar_prev::Tx = copy(xbar)
    d::Tx = similar(x)
    x_d::Tx = similar(x)
    gamma::R
    f_u::R
    g_v::R
    H::TH
    tau::R = zero(gamma)
    u0::Tx = similar(x)
    u1::Tx = similar(x)
    temp_x1::Tx = similar(x)
    temp_x2::Tx = similar(x)
end

function DRE(f_u::R, g_v, x, u, res, gamma) where {R}
    dot_product = R(0)
    for (x_i, u_i, res_i) in zip(x, u, res)
        dot_product += (x_i - u_i) * res_i
    end
    return f_u + g_v - real(dot_product) / gamma + 1 / (2 * gamma) * norm(res)^2
end

DRE(state::DRLSState) = DRE(state.f_u, state.g_v, state.x, state.u, state.res, state.gamma)

function Base.iterate(iter::DRLSIteration)
    x = copy(iter.x0)
    u, f_u = prox(iter.f, x, iter.gamma)
    w = 2 * u - x
    v, g_v = prox(iter.g, w, iter.gamma)
    res = u - v
    xbar = x - iter.lambda * res
    state = DRLSState(
        x = x,
        u = u,
        v = v,
        w = w,
        res = res,
        xbar = xbar,
        gamma = iter.gamma,
        f_u = f_u,
        g_v = g_v,
        H = initialize(iter.directions, x),
    )
    return state, state
end

function set_next_direction!(::QuasiNewtonStyle, ::DRLSIteration, state::DRLSState)
    mul!(state.d, state.H, state.res)
    state.d .*= -1
end
set_next_direction!(::NesterovStyle, ::DRLSIteration, state::DRLSState) =
    state.d .=
        iterate(state.H)[1] .* (state.xbar .- state.xbar_prev) .+ (state.xbar .- state.x)
set_next_direction!(::NoAccelerationStyle, ::DRLSIteration, state::DRLSState) =
    state.d .= state.xbar .- state.x
set_next_direction!(iter::DRLSIteration, state::DRLSState) =
    set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

function update_direction_state!(::QuasiNewtonStyle, ::DRLSIteration, state::DRLSState)
    state.res_prev .= state.res .- state.res_prev
    update!(state.H, state.d, state.res_prev)
end
update_direction_state!(::NesterovStyle, ::DRLSIteration, state::DRLSState) = return
update_direction_state!(::NoAccelerationStyle, ::DRLSIteration, state::DRLSState) = return
update_direction_state!(iter::DRLSIteration, state::DRLSState) =
    update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::DRLSIteration{R,Tx,Tf}, state::DRLSState) where {R,Tx,Tf}
    DRE_curr = DRE(state)
    threshold = iter.dre_sign * DRE_curr - iter.c / iter.gamma * norm(state.res)^2

    set_next_direction!(iter, state)

    state.x_d .= state.x .+ state.d
    state.xbar_prev, state.xbar = state.xbar, state.xbar_prev
    state.res_prev, state.res = state.res, state.res_prev
    state.tau = R(1)

    state.x .= state.x_d
    state.f_u = prox!(state.u, iter.f, state.x, iter.gamma)
    state.w .= 2 .* state.u .- state.x
    state.g_v = prox!(state.v, iter.g, state.w, iter.gamma)
    state.res .= state.u .- state.v
    state.xbar .= state.x .- iter.lambda * state.res

    update_direction_state!(iter, state)

    a, b, c = R(0), R(0), R(0)

    for k = 1:iter.max_backtracks
        if iter.dre_sign * DRE(state) <= threshold
            break
        end

        state.tau = k == iter.max_backtracks ? R(0) : state.tau / 2
        state.x .= state.tau .* state.x_d .+ (1 - state.tau) .* state.xbar_prev

        if ProximalCore.is_generalized_quadratic(Tf)
            if k == 1
                copyto!(state.u1, state.u)
                c = prox!(state.u0, iter.f, state.xbar_prev, iter.gamma)
                state.temp_x1 .= state.xbar_prev .- state.x_d
                state.temp_x2 .= state.xbar_prev .- state.u0
                b = real(dot(state.temp_x1, state.temp_x2)) / iter.gamma
                a = state.f_u - b - c
            end
            state.u .= state.tau .* state.u1 .+ (1 - state.tau) .* state.u0
            state.f_u = a * state.tau^2 + b * state.tau + c
        else
            state.f_u = prox!(state.u, iter.f, state.x, iter.gamma)
        end

        state.w .= 2 .* state.u .- state.x
        state.g_v = prox!(state.v, iter.g, state.w, iter.gamma)
        state.res .= state.u .- state.v
        state.xbar .= state.x .- iter.lambda * state.res
    end

    return state, state
end

default_stopping_criterion(tol, ::DRLSIteration, state::DRLSState) =
    norm(state.res, Inf) / state.gamma <= tol
default_solution(::DRLSIteration, state::DRLSState) = state.v
default_iteration_summary(it, ::DRLSIteration, state::DRLSState) =
    ("" => it,
    "f(u)" => state.f_u,
    "g(v)" => state.g_v,
    "γ" => state.gamma,
    "‖u - v‖/γ" => norm(state.res, Inf) / state.gamma,
    "τ" => state.tau)

"""
    DRLS(; <keyword-arguments>)

Constructs the Douglas-Rachford line-search algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

The returned object has type `IterativeAlgorithm{DRLSIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`DRLSIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `DRLSIteration` constructor upon call

# References
1. Themelis, Stella, Patrinos, "Douglas-Rachford splitting and ADMM for nonconvex optimization: Accelerated and Newton-type linesearch algorithms", Computational Optimization and Applications, vol. 82, no. 2, pp. 395-440 (2022).
"""
DRLS(;
    maxit = 1_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 10,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    DRLSIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:DRLSIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth,)),
    SimpleTerm(:g => (is_proximable,))
)
