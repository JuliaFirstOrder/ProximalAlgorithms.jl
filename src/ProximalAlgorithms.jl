module ProximalAlgorithms

using ProximalOperators
using AbstractOperators
using AbstractOperators.BlockArrays

import Base: start, next, done

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
    if verbose(solver) display(solver) end
    # NOTE: the following loop is translated into:
    #   it = start(solver)
    #   while !done(solver, it)
    #       (point, it) = next(solver, it)
    #       [...]
    #   end
    # See: https://docs.julialang.org/en/stable/manual/interfaces
    for (it, point) in enumerate(solver)
        if verbose(solver, it) display(solver, it) end
    end
    if verbose(solver) display(solver, it) end
    return (it, point)
end

# Functions `verbose` and `display` are used for inspecting the iterations.
# Here we provide their default behavior (no output).

verbose(sol::ProximalAlgorithm) = false
verbose(sol::ProximalAlgorithm, it) = false
function display(sol::ProximalAlgorithm) end
function display(sol::ProximalAlgorithm, it) end

# It remains to define what concrete ProximalAlgorithm types are and how
# `initialize`, `iterate`, `maxit`, `converged` work for each specific solver.
# This is done in the following included files.

include("algorithms/ForwardBackward.jl")
include("algorithms/ZeroFPR.jl")
include("algorithms/DouglasRachford.jl")

# include("algorithms/AsymmetricForwardBackwardAdjoint.jl")
# include("algorithms/VuCondat.jl")
# include("algorithms/ChambollePock.jl")
# include("algorithms/DavisYin.jl")

# The following template can be copy-pasted to implement new algorithms.

include("algorithms/Template.jl")

end # module
