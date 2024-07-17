using Test
using Aqua
using DifferentiationInterface
using ProximalAlgorithms

struct Quadratic{M,V}
    Q::M
    q::V
end

(f::Quadratic)(x) = dot(x, f.Q * x) / 2 + dot(f.q, x)

function ProximalAlgorithms.value_and_gradient(f::Quadratic, x)
    grad = f.Q * x + f.q
    return dot(grad, x) / 2 + dot(f.q, x), grad
end

@testset "Aqua" begin
    Aqua.test_all(ProximalAlgorithms; ambiguities=false)
end

include("utilities/test_ad.jl")
include("utilities/test_iteration_tools.jl")
include("utilities/test_fb_tools.jl")

include("accel/test_lbfgs.jl")
include("accel/test_anderson.jl")
include("accel/test_nesterov.jl")
include("accel/test_broyden.jl")

include("problems/test_equivalence.jl")
include("problems/test_elasticnet.jl")
include("problems/test_lasso_small.jl")
include("problems/test_lasso_small_strongly_convex.jl")
include("problems/test_linear_programs.jl")
include("problems/test_sparse_logistic_small.jl")
include("problems/test_nonconvex_qp.jl")
include("problems/test_verbose.jl")
