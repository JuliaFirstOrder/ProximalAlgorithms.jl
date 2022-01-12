# Themelis, Stella, Patrinos, "Douglas-Rachford splitting and ADMM
# for nonconvex optimization: Accelerated and Newton-type linesearch algorithms",
# arXiv:2005.10230, 2020.
#
# https://arxiv.org/abs/2005.10230

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

function drls_default_gamma(f, muf, Lf, alpha, lambda)
    if muf !== nothing && muf > 0
        return 1 / (alpha * muf)
    end
    if ProximalOperators.is_convex(f)
        return alpha / Lf
    else
        return alpha * (2 - lambda) / (2 * Lf)
    end
end

function drls_C(f, muf, Lf, gamma, lambda)
    a = muf === nothing || muf <= 0 ? gamma * Lf : 1 / (gamma * muf)
    m = ProximalOperators.is_convex(f) ? max(a - lambda / 2, 0) : 1
    return (lambda / ((1 + a)^2) * ((2 - lambda) / 2 - a * m))
end

"""
    DRLSIteration(; <keyword-arguments>)

Instantiate the Douglas-Rachford line-search algorithm (see [1]) for solving optimization problems
of the form

    minimize f(x) + g(x),

where `f` is smooth.

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `muf=nothing`: convexity modulus of f.
- `Lf=nothing`: Lipschitz constant of the gradient of f.
- `gamma`: stepsize to use, chosen appropriately based on Lf and muf by defaults.
- `max_backtracks=20`: maximum number of line-search backtracks.
- `directions=LBFGS(5)`: strategy to use to compute line-search directions.

# References
1. Themelis, Stella, Patrinos, "Douglas-Rachford splitting and ADMM for nonconvex optimization: Accelerated and Newton-type linesearch algorithms", arXiv:2005.10230, 2020.
"""
Base.@kwdef struct DRLSIteration{R,Tx,Tf,Tg,Tmuf,TLf,D}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    lambda::R = real(eltype(x0))(1)
    muf::Tmuf = nothing
    Lf::TLf = nothing
    gamma::R = drls_default_gamma(f, muf, Lf, alpha, lambda)
    c::R = beta * drls_C(f, muf, Lf, gamma, lambda)
    dre_sign::Int = muf === nothing || muf <= 0 ? 1 : -1
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

DRE(f_u::Number, g_v::Number, x, u, res, gamma) = f_u + g_v - real(dot(x - u, res)) / gamma + 1 / (2 * gamma) * norm(res)^2

DRE(state::DRLSState) = DRE(state.f_u, state.g_v, state.x, state.u, state.res, state.gamma)

function Base.iterate(iter::DRLSIteration)
    x = copy(iter.x0)
    u, f_u = prox(iter.f, x, iter.gamma)
    w = 2 * u - x
    v, g_v = prox(iter.g, w, iter.gamma)
    res = u - v
    xbar = x - iter.lambda * res
    state = DRLSState(
        x=x, u=u, v=v, w=w, res=res, xbar=xbar, gamma=iter.gamma, f_u=f_u,
        g_v=g_v, H=initialize(iter.directions, x),
    )
    return state, state
end

set_next_direction!(::QuasiNewtonStyle, ::DRLSIteration, state::DRLSState) = mul!(state.d, state.H, -state.res)
set_next_direction!(::NesterovStyle, ::DRLSIteration, state::DRLSState) = state.d .= iterate(state.H)[1] .* (state.xbar .- state.xbar_prev) .+ (state.xbar .- state.x)
set_next_direction!(::NoAccelerationStyle, ::DRLSIteration, state::DRLSState) = state.d .= state.xbar .- state.x
set_next_direction!(iter::DRLSIteration, state::DRLSState) = set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

update_direction_state!(::QuasiNewtonStyle, ::DRLSIteration, state::DRLSState) = update!(state.H, state.d, state.res - state.res_prev)
update_direction_state!(::NesterovStyle, ::DRLSIteration, state::DRLSState) = return
update_direction_state!(::NoAccelerationStyle, ::DRLSIteration, state::DRLSState) = return
update_direction_state!(iter::DRLSIteration, state::DRLSState) = update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::DRLSIteration{R}, state::DRLSState) where R
    DRE_curr = DRE(state)

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

    for k in 1:iter.max_backtracks
        if iter.dre_sign * DRE(state) <= iter.dre_sign * DRE_curr - iter.c / iter.gamma * norm(state.res_prev)^2
            break
        end

        state.tau = k == iter.max_backtracks ? R(0) : state.tau / 2
        state.x .= state.tau .* state.x_d .+ (1 - state.tau) .* state.xbar_prev

        if k == 1 && ProximalOperators.is_generalized_quadratic(iter.f)
            copyto!(state.u1, state.u)
            c = prox!(state.u0, iter.f, state.xbar_prev, iter.gamma)
            state.temp_x1 .= state.xbar_prev .- state.x_d
            state.temp_x2 .= state.xbar_prev .- state.u0
            b = real(dot(state.temp_x1, state.temp_x2)) / iter.gamma
            a = state.f_u - b - c
        end

        if ProximalOperators.is_generalized_quadratic(iter.f)
            state.u .= state.tau .* state.u1 .+ (1 - state.tau) .* state.u0
            state.f_u = a * state.tau ^ 2 + b * state.tau + c
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

# Solver

struct DRLS{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::DRLS)(x0; kwargs...)
    stop(state::DRLSState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it,
        state.gamma / state.gamma,
        norm(state.res, Inf),
        state.tau,
    )
    iter = DRLSIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.u, state_final.v, num_iters
end

DRLS(; maxit=1_000, tol=1e-8, verbose=false, freq=10, kwargs...) = 
    DRLS(maxit, tol, verbose, freq, kwargs)
