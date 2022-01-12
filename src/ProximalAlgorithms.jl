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

# utilities

include("utilities/ad.jl")
include("utilities/conjugate.jl")
include("utilities/fb_tools.jl")
include("utilities/iteration_tools.jl")

# acceleration operators

include("accel/traits.jl")
include("accel/lbfgs.jl")
include("accel/anderson.jl")
include("accel/nesterov.jl")
include("accel/broyden.jl")
include("accel/noaccel.jl")

# algorithms

include("algorithms/forward_backward.jl")
include("algorithms/fast_forward_backward.jl")
include("algorithms/zerofpr.jl")
include("algorithms/panoc.jl")
include("algorithms/douglas_rachford.jl")
include("algorithms/drls.jl")
include("algorithms/primal_dual.jl")
include("algorithms/davis_yin.jl")
include("algorithms/li_lin.jl")
include("algorithms/fista.jl")
include("algorithms/panocplus.jl")

end # module
