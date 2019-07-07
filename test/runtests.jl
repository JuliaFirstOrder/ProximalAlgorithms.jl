using Test

include("definitions/arraypartition.jl")

include("utilities/iterationtools.jl")
include("utilities/lbfgs.jl")
include("utilities/conjugate.jl")

include("problems/test_elasticnet.jl")
include("problems/test_lasso_small.jl")
include("problems/test_lasso_small_v_split.jl")
include("problems/test_lasso_small_h_split.jl")
include("problems/test_linear_programs.jl")
include("problems/test_sparse_logistic_small.jl")
include("problems/test_verbose.jl")
