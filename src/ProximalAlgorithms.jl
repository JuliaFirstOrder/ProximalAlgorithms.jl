module ProximalAlgorithms

using ProximalOperators
using AbstractOperators
using AbstractOperators.BlockArrays

import Base: start, next, done, println

include("utilities/identity.jl")
include("utilities/zero.jl")

abstract type ProximalAlgorithm{I, T} end

# The following methods give `ProximalAlgorithm` objects the iterable behavior.

function start(solver::ProximalAlgorithm{I, T})::I where {I, T}
    initialize(solver)
    return zero(I)
end

function next(solver::ProximalAlgorithm{I, T}, it::I)::Tuple{T, I} where {I, T}
    point::T = iterate(solver, it)
    return (point, it + one(I))
end

function done(solver::ProximalAlgorithm{I, T}, it::I)::Bool where {I, T}
    return it >= maxit(solver) || converged(solver, it)
end

# Running a `ProximalAlgorithm` unrolls the iterations

function run(solver::ProximalAlgorithm{I, T})::Tuple{I, T} where {I, T}
    local it, point
    # NOTE: the following loop is translated into:
    #
    # it = start(solver)
    # while !done(solver, it)
    #   (point, it) = next(solver, it)
    #   [...]
    # end
    #
    for (it, point) in enumerate(solver)
        if verbose(solver, it) display(solver, it) end
    end
    return (it, point)
end

# It remains to define what concrete ProximalAlgorithm types are and how
# `initialize`, `iterate`, `maxit`, `converged`, `verbose`, `display`
# work for each specific solver. This is done in the following included files.

include("algorithms/Template.jl")
include("algorithms/ForwardBackward.jl")
include("algorithms/ZeroFPR.jl")

# ...and then for example:

include("algorithms/DouglasRachford.jl")
# include("ChambollePock.jl")
# include("DavisYin.jl")

end # module
