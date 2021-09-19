@testset "Lasso small (h. split, $T)" for T in [Float32, Float64, ComplexF32, ComplexF64]
    using ProximalOperators
    using ProximalAlgorithms
    using LinearAlgebra
    using AbstractOperators: MatrixOp
    using RecursiveArrayTools: ArrayPartition

    A1 = T[
        1.0 -2.0 3.0
        2.0 -1.0 0.0
        -1.0 0.0 4.0
        -1.0 -1.0 -1.0
    ]
    A2 = T[
        -4.0 5.0
        -1.0 3.0
        -3.0 2.0
        1.0 3.0
    ]
    A = hcat(MatrixOp(A1), MatrixOp(A2))
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n1 = size(A1)
    _, n2 = size(A2)
    n = n1 + n2

    R = real(T)

    lam = R(0.1) * norm(A' * b, Inf)
    @test typeof(lam) == R

    f = Translate(SqrNormL2(R(1)), -b)
    g = SeparableSum(NormL1(lam), NormL1(lam))

    x_star = ArrayPartition(
        T[-3.877278911564627e-01, 0, 0],
        T[2.174149659863943e-02, 6.168435374149660e-01],
    )

    TOL = R(1e-4)

    @testset "ForwardBackward" begin

        ## Nonfast/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm([A1 A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150
        @test x0 == x0_backup

        # Nonfast/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300
        @test x0 == x0_backup

        # Fast/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm([A1 A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup

        # Fast/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver =
            ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
        @test x0 == x0_backup
    end

    @testset "ZeroFPR" begin

        # ZeroFPR/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm([A1 A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        # ZeroFPR/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
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

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g, Lf = opnorm([A1 A2])^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

        ## PANOC/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = solver(x0, f = f, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20
        @test x0 == x0_backup

    end

end
