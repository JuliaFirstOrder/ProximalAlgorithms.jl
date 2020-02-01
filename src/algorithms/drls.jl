# Douglas-Rachford line-search method

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

struct DRLS_iterable{R, C <: Union{R, Complex{R}}, Tx <: AbstractArray{C}, Tf, Tg, TH}
    f::Tf
    g::Tg
    x0::Tx
    lambda::R
    gamma::R
    c::R
    N::Int
    H::TH
end

mutable struct DRLS_state{R, Tx, TH}
    x::Tx
    u::Tx
    v::Tx
    w::Tx
    res::Tx
    xbar::Tx
    xbar_curr::Tx
    d::Tx
    x_d::Tx
    gamma::R
    f_u::R
    g_v::R
    H::TH
    tau::Maybe{R}
end

function DRE(state::DRLS_state)
    return (
        state.f_u + state.g_v - real(dot(state.x - state.u, state.res))/state.gamma
        + norm(state.res)^2/(2*state.gamma)
    )
end

function Base.iterate(iter::DRLS_iterable)
    x = iter.x0
    u, f_u = prox(iter.f, x, iter.gamma)
    w = 2 * u - x
    v, g_v = prox(iter.g, w, iter.gamma)
    res = u - v
    xbar = x - iter.lambda * res
    state = DRLS_state(
        x, u, v, w, res, xbar, zero(x), zero(x), zero(x), iter.gamma, f_u, g_v, iter.H, nothing
    )
    return state, state
end

function Base.iterate(iter::DRLS_iterable{R}, state::DRLS_state) where R
    DRE_curr = DRE(state)
    update!(iter.H, state.x, state.res)
    mul!(state.d, iter.H, -state.res)
    state.x_d .= state.x .+ state.d
    copyto!(state.xbar_curr, state.xbar)
    state.tau = R(1)
    state.x .= state.x_d
    for k in 1:iter.N
        state.f_u = prox!(state.u, iter.f, state.x, iter.gamma)
        state.w .= 2 .* state.u .- state.x
        state.g_v = prox!(state.v, iter.g, state.w, iter.gamma)
        state.res .= state.u .- state.v
        state.xbar .= state.x .- iter.lambda * state.res
        DRE_candidate = DRE(state)
        if DRE_candidate <= DRE_curr - iter.c * norm(state.res)^2
            return state, state
        end
        state.tau = state.tau / 2
        state.x .= state.tau .* state.x_d .+ (1 - state.tau) .* state.xbar_curr
    end
end

# Solver

struct DRLS{R}
    alpha::R
    beta::R
    gamma::Maybe{R}
    lambda::R
    N::Int
    memory::Int
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int

    function DRLS{R}(; alpha::R=R(0.95), beta::R=R(0.5),
        gamma::Maybe{R}=nothing, lambda::R=R(1), N::Int=20, memory::Int=5,
        maxit::Int=1000, tol::R=R(1e-8), verbose::Bool=false, freq::Int=10
    ) where R
        @assert 0 < alpha < 1
        @assert 0 < beta < 1
        @assert gamma === nothing || gamma > 0
        @assert 0 < lambda < 2
        @assert N > 0
        @assert memory >= 0
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(alpha, beta, gamma, lambda, N, memory, maxit, tol, verbose, freq)
    end
end

function (solver::DRLS{R})(
    x0::AbstractArray{C}; f=Zero(), g=Zero(), L::Maybe{R}=nothing
) where {R, C <: Union{R, Complex{R}}}
    stop(state::DRLS_state) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it, state.gamma / state.gamma, norm(state.res, Inf),
        (state.tau === nothing ? 0.0 : state.tau)
    )

    gamma = if solver.gamma === nothing && L !== nothing
        solver.alpha * (2 - solver.lambda) / (2 * L)
    else
        solver.gamma
    end

    C_gamma_lambda = solver.lambda / ((1 + gamma * L) ^ 2) * ((2 - solver.lambda) / (2 * gamma) - L)
    c = solver.beta * C_gamma_lambda

    iter = DRLS_iterable(
        f, g, x0, solver.lambda, gamma, c, solver.N, LBFGS(x0, solver.memory)
    )
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose iter = tee(sample(iter, solver.freq), disp) end

    num_iters, state_final = loop(iter)

    return state_final.u, state_final.v, num_iters
end

DRLS(::Type{R}; kwargs...) where R = DRLS{R}(; kwargs...)
DRLS(; kwargs...) = DRLS(Float64; kwargs...)
