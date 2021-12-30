using LinearAlgebra
using Test
using Random

using ProximalOperators
using ProximalAlgorithms

@testset "Lasso small (strongly convex, $T)" for T in [Float32, Float64]

    Random.seed!(777)
    dim = 5
    μf = T(1)
    Lf = T(10)

    x_star = convert(Vector{T}, 1.5 * rand(T, dim) .- 0.5)

    lam = (μf + Lf) / 2
    @test typeof(lam) == T

    D = Diagonal(sqrt(μf) .+ (sqrt(Lf) - sqrt(μf)) * rand(T, dim))
    D[1] = sqrt(μf)
    D[end] = sqrt(Lf)
    Q = qr(rand(T, (dim, dim))).Q
    A = Q * D * Q'
    b = A * x_star + lam * inv(A') * sign.(x_star)

    f = LeastSquares(A, b)
    h = NormL1(lam)

    TOL = T(1e-4)

    x0 = A \ b
    x0_backup = copy(x0)

    @testset "SFISTA" begin
        solver = ProximalAlgorithms.SFISTA(tol = TOL)
        y, it = solver(x0, f = f, h = h, Lf = Lf, μf = μf)
        @test eltype(y) == T
        @test norm(y - x_star) <= TOL
        @test it < 45
        @test x0 == x0_backup
    end

    @testset "ForwardBackward" begin
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        y, it = solver(x0, f = f, g = h, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 110
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward" begin
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        y, it = solver(x0, f = f, g = h, Lf = Lf, m = μf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 45
        @test x0 == x0_backup
    end

    @testset "DRLS" begin
        solver = ProximalAlgorithms.DRLS(tol = TOL)
        u, v, it = solver(x0, f = f, g = h, muf = μf)
        @test eltype(u) == T
        @test eltype(v) == T
        @test norm(v - x_star, Inf) <= TOL
        @test it < 14
        @test x0 == x0_backup
    end

    @testset "NOLIP" begin
        solver = ProximalAlgorithms.NOLIP(tol = TOL)
        y, it = solver(x0, f = f, g = h, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 45
        @test x0 == x0_backup
    end

end
