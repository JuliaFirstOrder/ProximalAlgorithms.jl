using Test

include("definitions/arraypartition.jl")
include("definitions/compose.jl")

include("utilities/iteration_tools.jl")
include("utilities/fb_tools.jl")
include("utilities/conjugate.jl")

include("accel/lbfgs.jl")
include("accel/anderson.jl")
include("accel/nesterov.jl")
include("accel/broyden.jl")
include("accel/noaccel.jl")

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
