using LinearAlgebra
using Test

using ProximalAlgorithms

@testset "Noaccel" begin
    L = ProximalAlgorithms.Noaccel()

    x = randn(10)
    y =  L*x

    @test y == x

    mul!(y, L, x)

    @test y == x
end
