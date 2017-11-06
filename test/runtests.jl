using ProximalAlgorithms
using Base.Test

@testset "ProximalAlgorithms" begin

@testset "Utilities" begin
    include("test_block.jl")
end

@testset "Algorithms" begin
    include("test_template.jl")
    include("test_lasso_small.jl")
    include("test_lasso_small_split_x.jl")
    include("test_lasso_small_split_f.jl")
end

end
