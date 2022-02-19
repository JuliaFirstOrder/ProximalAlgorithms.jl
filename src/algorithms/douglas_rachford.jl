# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

using Base.Iterators
using ProximalCore: Zero
using LinearAlgebra
using Printf

"""
    DouglasRachfordIteration(; <keyword-arguments>)

Iterator implementing the Douglas-Rachford splitting algorithm [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x).

See also: [`DouglasRachford`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: proximable objective term.
- `g=Zero()`: proximable objective term.
- `gamma`: stepsize to use.

# References
1. Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators", Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
"""
Base.@kwdef struct DouglasRachfordIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    gamma::R
end

Base.IteratorSize(::Type{<:DouglasRachfordIteration}) = Base.IsInfinite()

Base.@kwdef struct DouglasRachfordState{Tx}
    x::Tx
    y::Tx = similar(x)
    r::Tx = similar(x)
    z::Tx = similar(x)
    res::Tx = similar(x)
end

function Base.iterate(iter::DouglasRachfordIteration, state::DouglasRachfordState = DouglasRachfordState(x=copy(iter.x0)))
    prox!(state.y, iter.f, state.x, iter.gamma)
    state.r .= 2 .* state.y .- state.x
    prox!(state.z, iter.g, state.r, iter.gamma)
    state.res .= state.y .- state.z
    state.x .-= state.res
    return state, state
end

default_stopping_criterion(tol, iter::DouglasRachfordIteration, state::DouglasRachfordState) = norm(state.res, Inf) / iter.gamma <= tol
default_solution(::DouglasRachfordIteration, state::DouglasRachfordState) = state.y
default_display(it, iter::DouglasRachfordIteration, state::DouglasRachfordState) = @printf("%5d | %.3e\n", it, norm(state.res, Inf) / iter.gamma)

"""
    DouglasRachford(; <keyword-arguments>)

Constructs the Douglas-Rachford splitting algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x).

The returned object has type `IterativeAlgorithm{DouglasRachfordIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`DouglasRachfordIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `DouglasRachfordIteration` constructor upon call

# References
1. Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators", Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
"""
DouglasRachford(;
    maxit=1_000,
    tol=1e-8,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...
) = IterativeAlgorithm(DouglasRachfordIteration; maxit, stop, solution, verbose, freq, display, kwargs...)
