using Test
using Aqua
using AbstractDifferentiation
using ProximalAlgorithms

struct CustomBackend end

struct Quadratic{M, V}
    Q::M
    q::V
end

(f::Quadratic)(x) = dot(x, f.Q * x) / 2 + dot(f.q, x)

function ProximalAlgorithms.value_and_pullback_function(f::Quadratic, x)
    grad = f.Q * x + f.q
    return dot(grad, x) / 2 + dot(f.q, x), v -> (grad,)
end

@testset "Aqua" begin
    Aqua.test_all(ProximalAlgorithms; ambiguities=false)
end

include("definitions/arraypartition.jl")
include("definitions/compose.jl")

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
include("problems/test_lasso_small_v_split.jl")
include("problems/test_lasso_small_h_split.jl")
include("problems/test_linear_programs.jl")
include("problems/test_sparse_logistic_small.jl")
include("problems/test_nonconvex_qp.jl")
include("problems/test_verbose.jl")
