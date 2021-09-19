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
- `μ=0` : strong convexity constant of f.
- `L_f` : Lipschitz constant of ∇f.
- `λ` : stepsize to use; defaults to either 1 / (Lf - μ) if Lf > μ or 1 / (2 * Lf).
- `adaptive=false` : enables the use of adaptive stepsize selection.

# References
- [1] TODO
- [2] TODO
"""

@Base.kwdef struct FISTAIteration{R,Tx<:AbstractArray{R},Tf,Th}
    y0::Tx
    f::Tf = Zero()
    h::Th = Zero()
    Lf::R
    μ::R = 0.0
    adaptive::Bool = false # TODO: Implement adaptive FISTA.
end

Base.IteratorSize(::Type{<:FISTAIteration}) = Base.IsInfinite()

Base.@kwdef struct FISTAState{R, Tx}
    λ::R                       # stepsize (mutable if iter.adaptive == true)
    xt::Tx                     # prox center used to generate main iterate
    gradf_xt::Tx = zero(yPrev) # ∇f(xt)
    y:: Tx                     # main iterate
    yPrev::Tx                  # previous y
    gradf_y::Tx = zero(yPrev)  # ∇f(y)
    x::Tx                      # auxiliary iterate (see [TODO])
    xPrev::Tx                  # previous x
    τ::R                       # helper variable (see [TODO])
    a::R                       # helper variable (see [TODO])
    A::R                       # helper variable (see [TODO])
    APrev::R                   # previous A
end

function Base.iterate(iter::FISTAIteration,
                      state::FISTAState = FISTAState(yPrev=copy(iter.y0), xPrev=copy(iter.y0), APrev=0.0, λ=1/iter.Lf))
    # FISTA iteration.
    state.τ = state.λ * (1 + iter.μ * state.APrev)
    state.a = (state.τ + sqrt(state.τ ^ 2 + 4 * state.τ * state.APrev)) / 2
    state.A += state.a
    state.xt .= state.APrev / state.A * state.yPrev + state.a / state.A * state.xPrev
    gradient!(state.gradf_xt, iter.f, state.xt)
    λ2 = state.λ / (1 + state.λ * iter.μ)
    # FISTA acceleration step.
    prox!(state.y, iter.h, state.xt - λ2 * state.gradf_xt, λ2)
    gradient!(state.gradf_y, iter.f, state.y)
    state.x .+= state.a / (1 + state.A * iter.μ) * ((state.y - state.xt) / state.λ + iter.μ * (state.y - state.xPrev))
    # Update [*]Prev variables.
    state.yPrev .= state.y
    state.xPrev .= state.x
    state.APrev = state.A
    return state, state
end

## Solver.

struct FISTA{R, K}
    maxit::Int
    tol::R
    termination_type::String
    verbose::Bool
    freq::Int
    kwargs::K
end

# Different stopping conditions (sc). Returns the current residual value and whether or not a stopping condition holds.
function check_sc(state::FISTAState, iter::FISTAIteration, tol, termination_type)
    if termination_type == "AIPP"
        # See [TODO]. More specifically, r ∈ ∂_η(f + h)(y).
        r = (iter.y0 - state.x) / state.A
        η = (norm(iter.y0 - state.y) ^ 2 - norm(state.x - state.y) ^ 2) / (2 * state.A)
        res = (norm(r) ^ 2 + max(η, 0.0)) / max(norm(iter.y0 - state.y + r) ^ 2, 1e-16)
    else
        # Classic (approximate) first-order stationary point (see [TODO]). More specifically, r ∈ ∇f(y) + ∂h(y).
        r = state.gradf_y - state.gradf_xt + iter.Lf * (state.xt - state.y)
        res = norm(r)
    end
    return res, (res <= tol || res ≈ tol)
end

# Functor ('function-like object') for the above type.
function (solver::FISTA)(y0; kwargs...)
    iter = FISTAIteration(; y0=y0, solver.kwargs..., kwargs...)
    stop(state::FISTAState) = check_sc(state, iter, solver.tol, solver.termination_type)[2]
    disp((it, state)) = @printf("%5d | %.3e\n", it, check_sc(state, iter, solver.tol, solver.termination_type)[1])
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    num_iters, state_final = loop(iter)
    return state_final.y, num_iters
end

FISTA(; maxit=1_000, tol=1e-8, termination_type="", verbose=false, freq=100, kwargs...) =
    FISTA(maxit, tol, termination_type, verbose, freq, kwargs)

