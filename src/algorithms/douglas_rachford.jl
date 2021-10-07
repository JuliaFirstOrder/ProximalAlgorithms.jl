# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    DouglasRachfordIteration(; <keyword-arguments>)

Instantiate the Douglas-Rachford splitting algorithm (see [1]) for solving
convex optimization problems of the form

    minimize f(x) + g(x).

# Keyword arguments
- `x0`: initial point.
- `f=Zero()`: proximable objective term.
- `g=Zero()`: proximable objective term.
- `gamma`: stepsize to use.

# References
- [1] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
Optimization" (2008).
- [2] Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1,
pp. 183-202 (2009).
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
    y::Tx = zero(x)
    r::Tx = zero(x)
    z::Tx = zero(x)
    res::Tx = zero(x)
end

function Base.iterate(iter::DouglasRachfordIteration, state::DouglasRachfordState = DouglasRachfordState(x=copy(iter.x0)))
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
    iter = DouglasRachfordIteration(; x0=x0, solver.kwargs..., kwargs...)
    gamma = iter.gamma
    stop(state::DouglasRachfordState) = norm(state.res, Inf) / gamma <= solver.tol
    disp((it, state)) = @printf("%5d | %.3e\n", it, norm(state.res, Inf) / gamma)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.y, state_final.z, num_iters
end

DouglasRachford(; maxit=1_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    DouglasRachford(maxit, tol, verbose, freq, kwargs)
