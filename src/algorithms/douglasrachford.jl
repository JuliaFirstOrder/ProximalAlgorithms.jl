# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

@Base.kwdef struct DRS_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    gamma::R
end

Base.IteratorSize(::Type{<:DRS_iterable}) = Base.IsInfinite()

mutable struct DRS_state{Tx}
    x::Tx
    y::Tx
    r::Tx
    z::Tx
    res::Tx
end

DRS_state(iter::DRS_iterable) =
    DRS_state(copy(iter.x0), zero(iter.x0), zero(iter.x0), zero(iter.x0), zero(iter.x0))

function Base.iterate(iter::DRS_iterable, state::DRS_state = DRS_state(iter))
    prox!(state.y, iter.f, state.x, iter.gamma)
    state.r .= 2 .* state.y .- state.x
    prox!(state.z, iter.g, state.r, iter.gamma)
    state.res .= state.y .- state.z
    state.x .-= state.res
    return state, state
end

# Solver

struct DouglasRachford{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::DouglasRachford)(x0; kwargs...)
    iter = DRS_iterable(; x0=x0, solver.kwargs..., kwargs...)
    gamma = iter.gamma
    stop(state::DRS_state) = norm(state.res, Inf) / gamma <= solver.tol
    disp((it, state)) = @printf("%5d | %.3e\n", it, norm(state.res, Inf) / gamma)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.y, state_final.z, num_iters
end

# Outer constructors

"""
    DouglasRachford([gamma, maxit, tol, verbose, freq])

Instantiate the Douglas-Rachford splitting algorithm (see [1]) for solving
convex optimization problems of the form

    minimize f(x) + g(x),

If `solver = DouglasRachford(args...)`, then the above problem is solved with

    solver(x0, [f, g])

Optional keyword arguments:

* `gamma::Real` (default: `1.0`), stepsize parameter.
* `maxit::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

References:

[1] Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
Proximal Point Algorithm for Maximal Monotone Operators",
Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
"""
DouglasRachford(; maxit=1_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    DouglasRachford(maxit, tol, verbose, freq, kwargs)
