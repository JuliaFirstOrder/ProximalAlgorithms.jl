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

Base.@kwdef struct DRLSIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg,TH}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = begin
        if ProximalOperators.is_convex(f)
            alpha / Lf
        else
            alpha * (2 - lambda) / (2 * Lf)
        end
    end
    lambda::R = real(eltype(x0))(1)
    c::R = begin
        m = if ProximalOperators.is_convex(f)
            max(gamma * Lf - lambda / 2, 0)
        else
            1
        end
        C_gamma_lambda = (lambda / ((1 + gamma * Lf)^2) * ((2 - lambda) / 2 - gamma * Lf * m))
        c = beta * C_gamma_lambda
    end
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

function DRE(state::DRLSState)
    return (
        state.f_u + state.g_v - real(dot(state.x - state.u, state.res)) / state.gamma +
        1 / (2 * state.gamma) * norm(state.res)^2
    )
end

function Base.iterate(iter::DRLSIteration)
    x = copy(iter.x0)
    u, f_u = prox(iter.f, x, iter.gamma)
    w = 2 * u - x
    v, g_v = prox(iter.g, w, iter.gamma)
    res = u - v
    xbar = x - iter.lambda * res
    state = DRLSState(; x, u, v, w, res, xbar, iter.gamma, f_u, g_v, iter.H)
    return state, state
end

function Base.iterate(iter::DRLSIteration{R}, state::DRLSState) where {R}
    DRE_curr = DRE(state)

    mul!(state.d, iter.H, -state.res)
    state.x_d .= state.x .+ state.d
    copyto!(state.xbar_prev, state.xbar)
    copyto!(state.res_prev, state.res)
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

        if DRE_candidate <= DRE_curr - iter.c / iter.gamma * norm(state.res)^2
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
