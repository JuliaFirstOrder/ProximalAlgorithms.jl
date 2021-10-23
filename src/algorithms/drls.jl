# Themelis, Stella, Patrinos, "Douglas-Rachford splitting and ADMM
# for nonconvex optimization: Accelerated and Newton-type algorithms",
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

function drls_C(f, muf, Lf, gamma, lambda, beta)
    a = muf === nothing || muf <= 0 ? gamma * Lf : 1 / (gamma * muf)
    m = ProximalOperators.is_convex(f) ? max(a - lambda / 2, 0) : 1
    return (lambda / ((1 + a)^2) * ((2 - lambda) / 2 - a * m))
end

Base.@kwdef struct DRLSIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg,TH}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    lambda::R = real(eltype(x0))(1)
    muf::Maybe{R} = nothing
    Lf::Maybe{R} = nothing
    gamma::R = drls_default_gamma(f, muf, Lf, alpha, lambda)
    c::R = beta * drls_C(f, muf, Lf, gamma, lambda, beta)
    dre_sign::Int = muf === nothing || muf <= 0 ? 1 : -1
    max_backtracks::Int = 20
    H::TH = LBFGS(x0, 5)
end

Base.IteratorSize(::Type{<:DRLSIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct DRLSState{R,Tx,TH}
    x::Tx
    u::Tx
    v::Tx
    w::Tx
    res::Tx
    res_prev::Tx = zero(x)
    xbar::Tx
    xbar_prev::Tx = zero(x)
    d::Tx = zero(x)
    x_d::Tx = zero(x)
    gamma::R
    f_u::R
    g_v::R
    H::TH
    tau::Maybe{R} = nothing
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
        g_v=g_v, H=iter.H,
    )
    return state, state
end

function Base.iterate(iter::DRLSIteration{R}, state::DRLSState) where {R}
    DRE_curr = DRE(state)

    mul!(state.d, iter.H, -state.res)
    state.x_d .= state.x .+ state.d
    state.xbar_prev, state.xbar = state.xbar, state.xbar_prev
    state.res_prev, state.res = state.res, state.res_prev
    state.tau = R(1)
    state.x .= state.x_d

    for k = 1:iter.max_backtracks
        state.f_u = prox!(state.u, iter.f, state.x, iter.gamma)
        state.w .= 2 .* state.u .- state.x
        state.g_v = prox!(state.v, iter.g, state.w, iter.gamma)
        state.res .= state.u .- state.v

        if k == 1
            update!(iter.H, state.d, state.res - state.res_prev)
        end

        state.xbar .= state.x .- iter.lambda * state.res
        DRE_candidate = DRE(state)

        if iter.dre_sign * DRE_candidate <= iter.dre_sign * DRE_curr - iter.c / iter.gamma * norm(state.res_prev)^2
            return state, state
        end

        state.tau = state.tau / 2
        state.x .= state.tau .* state.x_d .+ (1 - state.tau) .* state.xbar_prev
    end

    @warn "stepsize `tau` became too small ($(state.tau)), stopping the iterations"
    return nothing
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
        (state.tau === nothing ? 0.0 : state.tau)
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
