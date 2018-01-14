using ProximalOperators
using ProximalAlgorithms
using AbstractOperators.BlockArrays
using Base.Test

@testset "ProximalAlgorithms" begin

@testset "Utilities" begin
    include("test_block.jl")
    include("test_conjugate.jl")
end

@testset "Algorithms" begin
    include("test_template.jl")
    include("test_lasso_small.jl")
    include("test_lasso_small_split_x.jl")
    include("test_lasso_small_split_f.jl")
    include("test_l1logreg_small.jl")
end

end
