using ProximalOperators
using ProximalAlgorithms
using AbstractOperators
using LinearAlgebra
using SparseArrays
using FFTW
using Random
using Test
using DelimitedFiles


@testset "ProximalAlgorithms" begin

@testset "Utilities" begin
    include("test_conjugate.jl")
end

@testset "Algorithms" begin
    include("test_template.jl")
    include("test_lasso_small.jl")
    include("test_lasso_harder.jl")
    include("test_lasso_small_split_x.jl")
    include("test_lasso_small_split_f.jl")
    include("test_l1logreg_small.jl")
    include("test_afba.jl")
    include("test_afba_LP.jl")
end

end
