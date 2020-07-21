module ProximalAlgorithms

const RealOrComplex{R} = Union{R, Complex{R}}
const Maybe{T} = Union{T, Nothing}

include("compat.jl")

# utilities

<<<<<<< HEAD
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
include("algorithms/AsymmetricForwardBackwardAdjoint.jl")

# include("algorithms/VuCondat.jl")
# include("algorithms/ChambollePock.jl")
# include("algorithms/DavisYin.jl")

# The following template can be copy-pasted to implement new algorithms.

include("algorithms/Template.jl")
=======
include("utilities/conjugate.jl")
include("utilities/fbetools.jl")
include("utilities/iterationtools.jl")

# acceleration operators

include("accel/lbfgs.jl")
include("accel/anderson.jl")
include("accel/nesterov.jl")
include("accel/broyden.jl")

# algorithms

include("algorithms/forwardbackward.jl")
include("algorithms/zerofpr.jl")
include("algorithms/panoc.jl")
include("algorithms/douglasrachford.jl")
include("algorithms/drls.jl")
include("algorithms/primaldual.jl")
include("algorithms/davisyin.jl")
include("algorithms/lilin.jl")
>>>>>>> upstreamPA/master

end # module
