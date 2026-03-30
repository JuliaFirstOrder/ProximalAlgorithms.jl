module ProximalAlgorithms

using ADTypes: ADTypes
using DifferentiationInterface: DifferentiationInterface
using ProximalCore
using ProximalCore: prox, prox!
using Printf

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T,Nothing}

"""
    AutoDifferentiable(f, backend)

Callable struct wrapping function `f` to be auto-differentiated using `backend`.

When called, it evaluates the same as `f`, while its gradient
is implemented using `backend` for automatic differentiation.
The backend can be any of those supported by [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl).
"""
struct AutoDifferentiable{F,B<:ADTypes.AbstractADType}
    f::F
    backend::B
end

(f::AutoDifferentiable)(x) = f.f(x)

"""
    value_and_gradient(f, x)

Return a tuple containing the value of `f` at `x` and the gradient of `f` at `x`.
"""
value_and_gradient

function value_and_gradient(f::AutoDifferentiable, x)
    return DifferentiationInterface.value_and_gradient(f.f, f.backend, x)
end

function value_and_gradient(f::ProximalCore.Zero, x)
    return f(x), zero(x)
end

# various utilities

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

struct IterativeAlgorithm{IteratorType,H,S,I,D,K}
    maxit::Int
    stop::H
    solution::S
    verbose::Bool
    freq::Int
    summary::I
    display::D
    kwargs::K
end

"""
    IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, summary, display, kwargs...)

Wrapper for an iterator type `T`, adding termination and verbosity options on top of it.

This is a conveniency constructor to allow for "partial" instantiation of an iterator of type `T`.
The resulting "algorithm" object `alg` can be called on a set of keyword arguments, which will be merged
to `kwargs` and passed on to `T` to construct an iterator which will be looped over.
Specifically, if an algorithm is constructed as

    alg = IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, summary, display, kwargs...)

then calling it with

    alg(; more_kwargs...)

will internally loop over an iterator constructed as

    T(; alg.kwargs..., more_kwargs...)

# Note
This constructor is not meant to be used directly: instead, algorithm-specific constructors
should be defined on top of it and exposed to the user, that set appropriate default functions
for `stop`, `solution`, `summary`, `display`.

# Arguments
* `T::Type`: iterator type to use
* `maxit::Int`: maximum number of iteration
* `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
* `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
* `verbose::Bool`: whether the algorithm state should be displayed
* `freq::Int`: every how many iterations to display the algorithm state
* `summary::Function`: function returning a summary of the iteration state, `summary(k::Int, iter::T, state)` should return a vector of pairs `(name, value)`
* `display::Function`: display function, `display(k::Int, alg, iter::T, state)` should display a summary of the iteration state
* `kwargs...`: keyword arguments to pass on to `T` when constructing the iterator
"""
IterativeAlgorithm(T; maxit, stop, solution, verbose, freq, summary, display, kwargs...) =
    IterativeAlgorithm{T,typeof(stop),typeof(solution),typeof(summary),typeof(display),typeof(kwargs)}(
        maxit,
        stop,
        solution,
        verbose,
        freq,
        summary,
        display,
        kwargs,
    )

function default_display(k, alg, iter, state, printfunc=println)
    if alg.freq > 0
        summary = alg.summary(k, iter, state)
        column_widths = map(pair -> max(length(pair.first), pair.second isa Integer ? 5 : 9), summary)
        if k == 0
            keys = map(first, summary)
            first_line = [_get_centered_text(key, width) for (width, key) in zip(column_widths, keys)]
            printfunc(join(first_line, " | "))
            second_line = [repeat('-', width) for width in column_widths]
            printfunc(join(second_line, "-|-"), "-")
        else
            values = map(last, summary)
            parts = [_format_value(value, width) for (width, value) in zip(column_widths, values)]
            printfunc(join(parts, " | "))
        end
    else
        summary = alg.summary(k, iter, state)
        if summary[1].first == ""
            summary = ("total iterations" => k, summary[2:end]...)
        end
        items = map(pair -> @sprintf("%s=%s", pair.first, _format_value(pair.second, 0)), summary)
        printfunc(join(items, ", "))
    end
end

function _get_centered_text(text, width)
    l = length(text)
    if l >= width
        return text
    end
    left_padding = div(width - l, 2)
    right_padding = width - l - left_padding
    return repeat(" ", left_padding) * text * repeat(" ", right_padding)
end

function _format_value(value, width)
    if value isa Integer
        return @sprintf("%*d", width, value)
    elseif value isa Float64 || value isa Float32
        return @sprintf("%*.3e", width, value)
    else
        return @sprintf("%*s", width, string(value))
    end
end

function (alg::IterativeAlgorithm{IteratorType})(; kwargs...) where {IteratorType}
    iter = IteratorType(; alg.kwargs..., kwargs...)
    for (k, state) in enumerate(iter)
        if k == 1 && alg.verbose && alg.freq > 0
            alg.display(0, alg, iter, state)
        end
        if k >= alg.maxit || alg.stop(iter, state)
            alg.verbose && alg.display(k, alg, iter, state)
            return (alg.solution(iter, state), k)
        end
        alg.verbose && alg.freq > 0 && mod(k, alg.freq) == 0 && alg.display(k, alg, iter, state)
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
