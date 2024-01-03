using ProximalOperators: LeastSquares, NormL1, SeparableSum, Sum, Translate
using ProximalAlgorithms
using LinearAlgebra
using AbstractOperators: MatrixOp

@testset "Lasso small (v. split, $T)" for T in [Float32, Float64, ComplexF32, ComplexF64]
    A1 = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
    ]
    A2 = T[
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    A = vcat(MatrixOp(A1), MatrixOp(A2))
    b1 = T[1.0, 2.0]
    b2 = T[3.0, 4.0]

    m1, n = size(A1)
    m2, _ = size(A2)
    m = m1 + m2

    R = real(T)

    lam = R(0.1) * norm([A1; A2]' * [b1; b2], Inf)
    @test typeof(lam) == R

    f = SeparableSum(Translate(SqrNormL2(R(1)), -b1), Translate(SqrNormL2(R(1)), -b2))
    f2 = Sum(LeastSquares(A1, b1), LeastSquares(A2, b2))
    g = NormL1(lam)

    Lf = opnorm([A1; A2])^2

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ForwardBackward" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        x, it = solver(x0 = x0, f = f2, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150
        @test x0 == x0_backup

        # Nonfast/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0 = x0, f = f2, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300
        @test x0 == x0_backup

        # Fast/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        x, it = solver(x0 = x0, f = f2, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup

        # Fast/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver =
            ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0 = x0, f = f2, g = g)
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
        x, it = solver(x0 = x0, f = f, A = A, g = g, Lf = opnorm([A1; A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        # ZeroFPR/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL)
        x, it = solver(x0 = x0, f = f, A = A, g = g)
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
        x, it = solver(x0 = x0, f = f, A = A, g = g, Lf = opnorm([A1; A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        ## PANOC/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = solver(x0 = x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

    end

    @testset "PANOCplus" begin

        # PANOCplus/Nonadaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(tol = TOL)
        x, it = solver(x0 = x0, f = f, A = A, g = g, Lf = opnorm([A1; A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        ## PANOCplus/Adaptive

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL)
        x, it = solver(x0 = x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

    end

end
