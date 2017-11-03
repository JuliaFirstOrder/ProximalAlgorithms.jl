module ProximalAlgorithms

using ProximalOperators
using AbstractOperators

import Base: start, next, done, println

include("utilities/identity.jl")
include("utilities/zero.jl")
include("utilities/block.jl")
include("utilities/broadcast.jl")

abstract type ProximalAlgorithm end

# The following methods give `ProximalAlgorithm` objects the iterable behavior.

function start(solver::ProximalAlgorithm)
    initialize(solver)
    return 0
end

function next(solver::ProximalAlgorithm, it::Tint) where {Tint}
    ret = iterate(solver, it)
    return (ret, it + one(Tint))
end

function done(solver::ProximalAlgorithm, it)
    return it >= maxit(solver) || converged(solver, it)
end

# Running a `ProximalAlgorithm` executes the iterations
# Records iter number separately from iterable mechanism so as to return them

function run(solver::ProximalAlgorithm)
    it = 0
    for (it, _) in enumerate(solver)
        if verbose(solver, it) display(solver, it) end
    end
    return it
end

# It remains to define what concrete ProximalAlgorithm types are and how
# `initialize`, `iterate`, `verbose`, `display`, `maxit`, `converged`
# work for each specific solver. This is done in the following included files.

include("algorithms/Template.jl")
include("algorithms/ForwardBackward.jl")

# ...and then for example:

# include("DouglasRachford.jl")
# include("ChambollePock.jl")
# include("DavisYin.jl")

end # module
