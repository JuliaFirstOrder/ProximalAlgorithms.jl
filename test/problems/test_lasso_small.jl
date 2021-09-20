using LinearAlgebra
using Test

using ProximalOperators
using ProximalAlgorithms

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

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ForwardBackward" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150
        @test x0 == x0_backup

        # Nonfast/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300
        @test x0 == x0_backup

        # Fast/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, fast = true)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup

        # Fast/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver =
            ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true, fast = true)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
        @test x0 == x0_backup
    end

    @testset "ZeroFPR" begin

        # ZeroFPR/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        # ZeroFPR/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

    end

    @testset "PANOC" begin

        # PANOC/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        ## PANOC/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

    end

    @testset "DouglasRachford" begin

        # Douglas-Rachford

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver =
            ProximalAlgorithms.DouglasRachford(gamma = R(10) / opnorm(A)^2, tol = TOL)
        y, z, it = solver(x0, f = f2, g = g)
        @test eltype(y) == T
        @test eltype(z) == T
        @test norm(y - x_star, Inf) <= TOL
        @test norm(z - x_star, Inf) <= TOL
        @test it < 30
        @test x0 == x0_backup

    end

    @testset "DouglasRachford line search" begin

        # Douglas-Rachford line search

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DRLS(tol = 10 * TOL)
        y, z, it = solver(x0, f = f2, g = g, Lf = opnorm(A)^2)
        @test eltype(y) == T
        @test eltype(z) == T
        @test norm(y - x_star, Inf) <= 10 * TOL
        @test norm(z - x_star, Inf) <= 10 * TOL
        @test it < 26
        @test x0 == x0_backup

    end

    @testset "AFBA" begin

        x0 = zeros(T, n)
        x0_backup = copy(x0)

        solver = ProximalAlgorithms.AFBA(theta = R(1), mu = R(1), tol = R(1e-6))
        x_afba, y_afba, it_afba = solver(x0, zeros(T, n), f = f2, g = g, beta_f = opnorm(A)^2)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 80
        @test x0 == x0_backup

        solver = ProximalAlgorithms.AFBA(theta = R(1), mu = R(1), tol = R(1e-6))
        x_afba, y_afba, it_afba = solver(x0, zeros(T, n), f = f2, h = g, beta_f = opnorm(A)^2)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 100
        @test x0 == x0_backup

        solver = ProximalAlgorithms.AFBA(theta = R(1), mu = R(1), tol = R(1e-6))
        x_afba, y_afba, it_afba = solver(x0, zeros(T, m), h = f, L = A, g = g)
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= 150
        @test x0 == x0_backup

    end

    @testset "FISTA" begin

        # FISTA

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FISTA(tol = 10 * TOL)
        y, it = solver(x0, f = f2, h = g, Lf = opnorm(A)^2)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= 10 * TOL
        @test it < 100
        @test x0 == x0_backup

    end

end
