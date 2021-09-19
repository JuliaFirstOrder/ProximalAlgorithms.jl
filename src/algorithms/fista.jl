# TODO: Add the right citation here.

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    FistaIteration(; <keyword-arguments>)

Instantiate the FISTA splitting algorithm (see [TODO]) for solving convex optimization problems of the form

    minimize f(x) + h(x).

# Arguments
- `y0`: accelerated iterate in the domain of h.
- `x0`: auxiliary iterate (see [TODO]).
- `f=Zero()`: smooth objective term.
- `h=Zero()`: proximable objective term.
- `λ` : stepsize to use.
- `μ` : strong convexity constant of f.
- `A0` : parameter used in the acceleration step (see [TODO]).

# References
- [1] TODO
- [2] TODO
"""

@Base.kwdef struct FISTAIteration{R,Tx<:AbstractArray{R},Tf,Th}
    y0::Tx
    f::Tf = Zero()
    h::Tg = Zero()
    λ::R
    μ::R = 0.0
    adaptive::Bool = false # TODO: Implement adaptive FISTA.
end

Base.IteratorSize(::Type{<:FISTAIteration}) = Base.IsInfinite()

Base.@kwdef struct FISTAState{R, Tx}
    # Auxiliary iterates.
    τ::R
    a::R
    A::R
    xt::Tx
    # Storage.
    gradf::Tx = zero(yPrev)
    # Variables forming the main inclusion [r ∈ ∂_η(f+h)(y)]
    y::Tx
    r::Tx = zero(y)
    η::R = 0.0
    # Record variables.
    yPrev::Tx
    xPrev::Tx
    APrev::R
end

function Base.iterate(iter::FISTAIteration,
                      state::FISTAState = FISTAState(yPrev=copy(iter.y0), xPrev=copy(iter.y0), APrev=0.0))
    # FISTA iteration.
    state.τ = iter.λ * (1 + iter.μ * state.APrev)
    state.a = (state.τ + sqrt(state.τ ^ 2 + 4 * state.τ * state.APrev)) / 2
    state.A += state.a
    state.xt .= state.APrev / state.A * state.yPrev + state.a / state.A * state.xPrev
    gradient!(state.gradf, iter.f, state.xt)
    λ2 = iter.λ / (1 + iter.λ * iter.μ)
    # FISTA acceleration step.
    prox!(state.y, iter.h, state.xt - λ2 * state.gradf, λ2)
    state.x .+= state.a / (1 + state.A * iter.μ) * ((state.y - state.xt) / iter.λ + iter.μ * (state.y - state.xPrev))
    state.r = (iter.y0 - state.x) / state.A
    state.η = (norm(iter.y0 - state.y) ^ 2 - norm(state.x - state.y) ^ 2) / (2 * state.A)
    # Update [*]Prev variables.
    state.yPrev .= state.y
    state.xPrev .= state.x
    state.APrev = state.A
    return state, state
end

## Solver.

struct FISTA{R, K}
    maxit::Int
    tol::R # See [TODO].
    type::String
    verbose::Bool
    freq::Int
    kwargs::K
end

# Alias for the above struct.
FISTA(; maxit=1_000, tol=1e-8, verbose=false, freq=100, type="", kwargs...) =
    FISTA(maxit, tol, verbose, freq, type, kwargs)

# Main caller.
function (solver::FISTA)(y0; kwargs...)
    iter = FISTAIteration(; y0=y0, solver.kwargs..., kwargs...)
    if solver.type == "AIPP"
        # Relevant for the implementation of AIPP.
        stop(state::FISTAState) = norm(state.r) ^ 2 + state.η <= solver.tol * norm(state.y00 - state.y + state.r)
    else
        stop(state::FISTAState) = max(norm(state.r), state.η) <= solver.tol
    end
    disp((it, state)) = @printf("%5d | %.3e\n", it, (norm(state.r) ^ 2 + state.η) / norm(state.y00 - state.y + state.r))
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    num_iters, state_final = loop(iter)
    return state_final.y, state_final.r, state_final.η, num_iters
end

