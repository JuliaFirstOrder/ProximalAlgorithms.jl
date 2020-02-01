module ProximalAlgorithms

const RealOrComplex{R} = Union{R, Complex{R}}
const Maybe{T} = Union{T, Nothing}

include("compat.jl")

# utilities

include("utilities/conjugate.jl")
include("utilities/fbetools.jl")
include("utilities/iterationtools.jl")

# acceleration operators

include("accel/lbfgs.jl")
include("accel/anderson.jl")
include("accel/nesterov.jl")

# algorithms

include("algorithms/forwardbackward.jl")
include("algorithms/zerofpr.jl")
include("algorithms/panoc.jl")
include("algorithms/douglasrachford.jl")
include("algorithms/primaldual.jl")
include("algorithms/davisyin.jl")
include("algorithms/lilin.jl")

end # module
