using LinearAlgebra
using Test
using Random

using ProximalOperators
using ProximalAlgorithms

@testset "Lasso small strongly convex ($T)" for T in [Float32, Float64]

    Random.seed!(777)
    dim = 10

    D = Diagonal(T(1.0) .+ T(9.0) .* rand(T, dim))
    Q = qr(rand(T, (dim, dim))).Q

    A = convert(Matrix{T}, Q * D * Q')
    b = A * (T(2.0) * rand(T, dim) .- T(1.0))
    σ = svd(A).S
    m, n = size(A)

    # lam = T(0.1) * norm(A' * b, Inf)
    x_ls = A \ b
    lam = maximum(x_ls) / T(2.0)
    @test typeof(lam) == T

    f = LeastSquares(A, b)
    h = NormL1(lam)

    x_star = sign.(x_ls) .* max.(T(0.0), abs.(x_ls) .- lam)

    TOL = T(1e-6)

    @testset "SFISTA" begin

        # SFISTA

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.SFISTA(tol = TOL)
        y, it = solver(x0, f = f, h = h, Lf = maximum(σ)^2, μf = minimum(σ)^2)
        @test eltype(y) == T
        @test norm(y - x_star) <= TOL
        @test it < 100
        @test x0 == x0_backup

    end

end
