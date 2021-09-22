using LinearAlgebra
using Test
using Random

using ProximalOperators
using ProximalAlgorithms

@testset "Lasso small (strongly convex, $T)" for T in [Float32, Float64]

    Random.seed!(777)
    dim = 5
    μf = T(1.0)
    Lf = T(10.0)

    x_star = convert(Vector{T}, 1.5 * rand(T, dim) .- 0.5)

    lam = convert(T, (μf + Lf) / 2.0)
    @test typeof(lam) == T

    D = Diagonal(convert(Vector{T}, sqrt(μf) .+ (sqrt(Lf) - sqrt(μf)) * rand(T, dim)))
    D[1] = sqrt(μf)
    D[end] = sqrt(Lf)
    Q = qr(rand(T, (dim, dim))).Q
    A = convert(Matrix{T}, Q * D * Q')
    b = A * x_star + lam * inv(A') * sign.(x_star)

    f = LeastSquares(A, b)
    h = NormL1(lam)

    TOL = T(1e-4)

    @testset "SFISTA" begin

        # SFISTA

        x0 =  A \ b
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.SFISTA(tol = TOL)
        y, it = solver(x0, f = f, h = h, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star) <= TOL
        @test it < 200
        @test x0 == x0_backup

    end

end
