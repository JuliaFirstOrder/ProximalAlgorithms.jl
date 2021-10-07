using LinearAlgebra
using Test

using ProximalOperators
using ProximalAlgorithms

@testset "Lasso small (high accuracy)" for T in [Float64]
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

    TOL = 100 * eps(R)

    @testset "ForwardBackward" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 450
        @test x0 == x0_backup

        # Nonfast/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 785
        @test x0 == x0_backup

        # Fast/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, fast = true)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 610
        @test x0 == x0_backup

        # Fast/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver =
            ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true, fast = true)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 1055
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
        @test it < 21
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
        @test it < 50
        @test x0 == x0_backup

        ## PANOC/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 50
        @test x0 == x0_backup

    end

end