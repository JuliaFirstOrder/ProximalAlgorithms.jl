using LinearAlgebra
using Test

using ProximalOperators
using ProximalAlgorithms
using ProximalAlgorithms:
    LBFGS, Broyden, AndersonAcceleration,
    NesterovExtrapolation, FixedNesterovSequence, SimpleNesterovSequence

@testset "Lasso small ($T)" for T in [Float32, Float64, ComplexF32, ComplexF64]
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    lam = R(0.1) * norm(A' * b, Inf)
    @test typeof(lam) == R

    f = Translate(SqrNormL2(R(1)), -b)
    f2 = LeastSquares(A, b)
    g = NormL1(lam)

    Lf = opnorm(A)^2

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ForwardBackward (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        x, it = @inferred solver(x0, f = f2, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150
        @test x0 == x0_backup
    end

    @testset "ForwardBackward (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = @inferred solver(x0, f = f2, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        x, it = @inferred solver(x0, f = f2, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver =
            ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
        x, it = @inferred solver(x0, f = f2, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
        @test x0 == x0_backup
    end

    @testset "ZeroFPR (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(tol = TOL)
        x, it = @inferred solver(x0, f = f, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "ZeroFPR (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL)
        x, it = @inferred solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "PANOC (fixed step)" begin

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(tol = TOL)
        x, it = @inferred solver(x0, f = f, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    
    end

    @testset "PANOC (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = @inferred solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "PANOCplus (fixed step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(tol = TOL)
        x, it = @inferred solver(x0, f = f, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "PANOCplus (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL)
        x, it = @inferred solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup
    end

    @testset "DouglasRachford" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DouglasRachford(gamma = R(10) / opnorm(A)^2, tol = TOL)
        y, z, it = @inferred solver(x0, f = f2, g = g)
        @test eltype(y) == T
        @test eltype(z) == T
        @test norm(y - x_star, Inf) <= TOL
        @test norm(z - x_star, Inf) <= TOL
        @test it < 30
        @test x0 == x0_backup
    end

    @testset "DouglasRachford line search ($acc)" for (acc, maxit) in [
        (LBFGS(5), 17),
        (Broyden(), 19),
        (AndersonAcceleration(5), 12),
        (NesterovExtrapolation(FixedNesterovSequence), 36),
        (NesterovExtrapolation(SimpleNesterovSequence), 36),
    ]
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DRLS(tol = 10 * TOL, directions=acc)
        y, z, it = @inferred solver(x0, f = f2, g = g, Lf = Lf)
        @test eltype(y) == T
        @test eltype(z) == T
        @test norm(y - x_star, Inf) <= 10 * TOL
        @test norm(z - x_star, Inf) <= 10 * TOL
        @test it < maxit
        @test x0 == x0_backup
    end

    @testset "AFBA" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.AFBA(theta = R(1), mu = R(1), tol = R(1e-6))
        x_afba, y_afba, it_afba = @inferred solver(x0, zeros(T, n), f = f2, g = g, beta_f = opnorm(A)^2)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 80
        @test x0 == x0_backup

        solver = ProximalAlgorithms.AFBA(theta = R(1), mu = R(1), tol = R(1e-6))
        x_afba, y_afba, it_afba = @inferred solver(x0, zeros(T, n), f = f2, h = g, beta_f = opnorm(A)^2)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 100
        @test x0 == x0_backup

        solver = ProximalAlgorithms.AFBA(theta = R(1), mu = R(1), tol = R(1e-6))
        x_afba, y_afba, it_afba = @inferred solver(x0, zeros(T, m), h = f, L = A, g = g)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 150
        @test x0 == x0_backup
    end

    @testset "SFISTA" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.SFISTA(tol = 10 * TOL)
        y, it = @inferred solver(x0, f = f2, h = g, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= 10 * TOL
        @test it < 100
        @test x0 == x0_backup
    end

end
