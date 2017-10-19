module ProximalAlgorithms

using ProximalOperators
using AbstractOperators

import Base: start, next, done, println

abstract type ProximalAlgorithm end

# The following methods give `ProximalAlgorithm` objects the iterable behavior.

function start(solver::ProximalAlgorithm)
    initialize(solver)
    return 0
end

function next(solver::ProximalAlgorithm, it::Tint) where {Tint}
    ret = iterate(solver, it)
    if verbose(solver, it) display(it, solver) end
    return (ret, it+one(Tint))
end

function done(solver::ProximalAlgorithm, it)
    return it >= maxit(solver) || converged(solver, it)
end

# The following is to run a (previously constructed) solver

function run(solver::ProximalAlgorithm)
    it = 0
    for _ in solver it += 1 end
    return it
end

# It remains to define what concrete ProximalAlgorithm types are and how
# `initialize`, `iterate`, `verbose`, `display`, `maxit`, `converged`
# work for each specific solver. This is done in the following included files.

include("ForwardBackward.jl")

# ...and then for example:

# include("DouglasRachford.jl")
# include("ChambollePock.jl")
# include("DavisYin.jl")

end # module
