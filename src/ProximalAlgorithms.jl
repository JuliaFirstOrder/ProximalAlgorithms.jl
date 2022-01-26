module ProximalAlgorithms

using ProximalOperators
import ProximalOperators: prox!, gradient!

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

"""
    prox!(y, f, x, gamma)

Compute the proximal mapping of `f` at `x`, with stepsize `gamma`, and store the result in `y`.
Return the value of `f` at `y`.
"""
prox!(y, f, x, gamma)

"""
    gradient!(g, f, x)

Compute the gradient of `f` at `x`, and stores it in `y`. Return the value of `f` at `x`.
"""
gradient!(y, f, x)

# TODO move out
ProximalOperators.is_quadratic(::Any) = false

# various utilities

include("utilities/ad.jl")
include("utilities/conjugate.jl")
include("utilities/fb_tools.jl")
include("utilities/iteration_tools.jl")

# acceleration utilities

include("accel/traits.jl")
include("accel/lbfgs.jl")
include("accel/anderson.jl")
include("accel/nesterov.jl")
include("accel/broyden.jl")
include("accel/noaccel.jl")

# algorithm interface

struct IterativeAlgorithm{IteratorType, H, S, D, K}
    maxit::Int
    stop::H
    solution::S
    verbose::Bool
    freq::Int
    display::D
    kwargs::K
end

"""
    IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...)

Wrapper for an iterator type `T`, adding termination and verbosity options on top of it.

This is a conveniency constructor to allow for "partial" instantiation of an iterator of type `T`.
The resulting object `alg` can be called on a set of keyword arguments, which will be merged
to `kwargs` and passed on to `T` to construct an iterator which will be looped over.

# Note
This constructor is not meant to be used directly: instead, algorithm-specific constructors
should be defined on top of it and exposed to the user.

# Arguments
* `T::Type`: iterator type to use
* `maxit::Int`: maximum number of iteration
* `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
* `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
* `verbose::Bool`: whether the algorithm state should be displayed
* `freq::Int`: every how many iterations to display the algorithm state
* `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
* `kwargs`: keyword arguments to pass on to `T` when constructing the iterator
"""
IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, display, kwargs...) =
    IterativeAlgorithm{T, typeof(stop), typeof(solution), typeof(display), typeof(kwargs)}(
        maxit, stop, solution, verbose, freq, display, kwargs
    )

function (alg::IterativeAlgorithm{IteratorType})(; kwargs...) where IteratorType
    iter = IteratorType(; alg.kwargs..., kwargs...)
    for (k, state) in enumerate(iter)
        if k >= alg.maxit || alg.stop(iter, state)
            alg.verbose && alg.display(k, iter, state)
            return (alg.solution(iter, state), k)
        end
        alg.verbose && mod(k, alg.freq) == 0 && alg.display(k, iter, state)
    end
end

# algorithm implementations

include("algorithms/forward_backward.jl")
include("algorithms/fast_forward_backward.jl")
include("algorithms/zerofpr.jl")
include("algorithms/panoc.jl")
include("algorithms/douglas_rachford.jl")
include("algorithms/drls.jl")
include("algorithms/primal_dual.jl")
include("algorithms/davis_yin.jl")
include("algorithms/li_lin.jl")
include("algorithms/sfista.jl")
include("algorithms/panocplus.jl")

end # module
