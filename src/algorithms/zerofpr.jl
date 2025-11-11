# Themelis, Stella, Patrinos, "Forward-backward envelope for the sum of two
# nonconvex functions: Further properties and nonmonotone line-search
# algorithms", SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274–2303
# (2018).

"""
    ZeroFPRIteration(; <keyword-arguments>)

Iterator implementing the ZeroFPR algorithm [1].

This iterator solves optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).

See also: [`ZeroFPR`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `A=I`: linear operator (e.g. a matrix).
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of x ↦ f(Ax).
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).
- `adaptive=false`: forces the method stepsize to be adaptively adjusted.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `max_backtracks=20`: maximum number of line-search backtracks.
- `directions=LBFGS(5)`: strategy to use to compute line-search directions.

# References
1. Themelis, Stella, Patrinos, "Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms", SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274-2303 (2018).
"""
Base.@kwdef struct ZeroFPRIteration{R,Tx,Tf,TA,Tg,TLf,Tgamma,D}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (alpha / Lf)
    adaptive::Bool = gamma === nothing
    minimum_gamma::R = real(eltype(x0))(1e-7)
    max_backtracks::Int = 20
    directions::D = LBFGS(5)
end

Base.IteratorSize(::Type{<:ZeroFPRIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct ZeroFPRState{R,Tx,TAx,TH}
    x::Tx             # iterate
    Ax::TAx           # A times x
    f_Ax::R           # value of smooth term
    grad_f_Ax::TAx    # gradient of f at Ax
    At_grad_f_Ax::Tx  # gradient of smooth term
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    xbar::Tx          # forward-backward point
    g_xbar::R         # value of nonsmooth term (at xbar)
    res::Tx           # fixed-point residual at iterate (= x - xbar)
    H::TH             # variable metric
    tau::R = zero(gamma)
    Axbar::TAx = similar(Ax)
    grad_f_Axbar::TAx = similar(Ax)
    At_grad_f_Axbar::Tx = similar(x)
    xbarbar::Tx = similar(x)
    res_xbar::Tx = similar(x)
    xbar_prev::Tx = similar(x)
    res_xbar_prev::Tx = similar(x)
    is_prev_set::Bool = false
    d::Tx = similar(x)
    Ad::TAx = similar(Ax)
end

f_model(iter::ZeroFPRIteration, state::ZeroFPRState) =
    f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::ZeroFPRIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    f_Ax, grad_f_Ax = value_and_gradient(iter.f, Ax)
    gamma =
        iter.gamma === nothing ?
        iter.alpha / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax) :
        iter.gamma
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    xbar, g_xbar = prox(iter.g, y, gamma)
    state = ZeroFPRState(
        x = x,
        Ax = Ax,
        f_Ax = f_Ax,
        grad_f_Ax = grad_f_Ax,
        At_grad_f_Ax = At_grad_f_Ax,
        gamma = gamma,
        y = y,
        xbar = xbar,
        g_xbar = g_xbar,
        res = x - xbar,
        H = initialize(iter.directions, x),
    )
    return state, state
end

function set_next_direction!(::QuasiNewtonStyle, ::ZeroFPRIteration, state::ZeroFPRState)
    mul!(state.d, state.H, state.res_xbar)
    state.d .*= -1
end
set_next_direction!(::NoAccelerationStyle, ::ZeroFPRIteration, state::ZeroFPRState) =
    state.d .= .-state.res
set_next_direction!(iter::ZeroFPRIteration, state::ZeroFPRState) =
    set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

function update_direction_state!(
    ::QuasiNewtonStyle,
    ::ZeroFPRIteration,
    state::ZeroFPRState,
)
    state.xbar_prev .= state.xbar .- state.xbar_prev
    state.res_xbar_prev .= state.res_xbar .- state.res_xbar_prev
    update!(state.H, state.xbar_prev, state.res_xbar_prev)
end
update_direction_state!(::NoAccelerationStyle, ::ZeroFPRIteration, state::ZeroFPRState) =
    return
update_direction_state!(iter::ZeroFPRIteration, state::ZeroFPRState) =
    update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

reset_direction_state!(::QuasiNewtonStyle, ::ZeroFPRIteration, state::ZeroFPRState) =
    reset!(state.H)
reset_direction_state!(::NoAccelerationStyle, ::ZeroFPRIteration, state::ZeroFPRState) =
    return
reset_direction_state!(iter::ZeroFPRIteration, state::ZeroFPRState) =
    reset_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::ZeroFPRIteration{R}, state::ZeroFPRState) where {R}
    f_Axbar_upp, f_Axbar = if iter.adaptive == true
        gamma_prev = state.gamma
        state.gamma, state.g_xbar, f_Axbar, f_Axbar_upp = backtrack_stepsize!(
            state.gamma,
            iter.f,
            iter.A,
            iter.g,
            state.x,
            state.f_Ax,
            state.At_grad_f_Ax,
            state.y,
            state.xbar,
            state.g_xbar,
            state.res,
            state.Axbar,
            state.grad_f_Axbar,
            alpha = iter.alpha,
            minimum_gamma = iter.minimum_gamma,
        )
        if state.gamma != gamma_prev
            reset_direction_state!(iter, state)
        end
        f_Axbar_upp, f_Axbar
    else
        mul!(state.Axbar, iter.A, state.xbar)
        f_Axbar, grad_f_Axbar = value_and_gradient(iter.f, state.Axbar)
        state.grad_f_Axbar .= grad_f_Axbar
        f_model(iter, state), f_Axbar
    end

    # compute FBE
    FBE_x = f_Axbar_upp + state.g_xbar

    # compute residual at xbar
    mul!(state.At_grad_f_Axbar, iter.A', state.grad_f_Axbar)
    state.y .= state.xbar .- state.gamma .* state.At_grad_f_Axbar
    g_xbarbar = prox!(state.xbarbar, iter.g, state.y, state.gamma)
    state.res_xbar .= state.xbar .- state.xbarbar

    if state.is_prev_set == true
        update_direction_state!(iter, state)
    end

    copyto!(state.xbar_prev, state.xbar)
    copyto!(state.res_xbar_prev, state.res_xbar)
    state.is_prev_set = true

    set_next_direction!(iter, state)

    # Perform line-search over the FBE
    state.tau = R(1)
    mul!(state.Ad, iter.A, state.d)

    sigma = iter.beta * (0.5 / state.gamma) * (1 - iter.alpha)
    tol = 10 * eps(R) * (1 + abs(FBE_x))
    threshold = FBE_x - sigma * norm(state.res)^2 + tol

    for k = 1:iter.max_backtracks
        state.x .= state.xbar_prev .+ state.tau .* state.d
        state.Ax .= state.Axbar .+ state.tau .* state.Ad
        # TODO: can precompute most of next line in case f is quadratic
        state.f_Ax, grad_f_Ax = value_and_gradient(iter.f, state.Ax)
        state.grad_f_Ax .= grad_f_Ax
        mul!(state.At_grad_f_Ax, iter.A', state.grad_f_Ax)
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_xbar = prox!(state.xbar, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.xbar
        FBE_x = f_model(iter, state) + state.g_xbar

        if FBE_x <= threshold
            break
        end

        state.tau = k >= iter.max_backtracks - 1 ? R(0) : state.tau / 2
    end

    return state, state
end

default_stopping_criterion(tol, ::ZeroFPRIteration, state::ZeroFPRState) =
    norm(state.res, Inf) / state.gamma <= tol
default_solution(::ZeroFPRIteration, state::ZeroFPRState) = state.xbar
default_iteration_summary(it, ::ZeroFPRIteration, state::ZeroFPRState) =
    ("" => it,
    "f(Ax)" => state.f_Ax,
    "g(x̄)" => state.g_xbar,
    "γ" => state.gamma,
    "‖x - x̄‖/γ" => norm(state.res, Inf) / state.gamma,
    "τ" => state.tau,
    )

"""
    ZeroFPR(; <keyword-arguments>)

Constructs the ZeroFPR algorithm [1].

This algorithm solves optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).

The returned object has type `IterativeAlgorithm{ZeroFPRIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`ZeroFPRIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=10`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `ZeroFPRIteration` constructor upon call

# References
1. Themelis, Stella, Patrinos, "Forward-backward envelope for the sum of two nonconvex functions: Further properties and nonmonotone line-search algorithms", SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274-2303 (2018).
"""
ZeroFPR(;
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
    ZeroFPRIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:ZeroFPRIteration}) = AssumptionGroup(
    OperatorTerm(:f => (is_smooth,), :A => (is_linear,)),
    SimpleTerm(:g => (is_proximable,)),
)
