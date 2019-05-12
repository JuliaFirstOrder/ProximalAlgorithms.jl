@testset "Lasso small (h. split, $T)" for T in [Float32, Float64, ComplexF32, ComplexF64]

    using ProximalOperators
    using ProximalAlgorithms
    using LinearAlgebra
    using AbstractOperators: MatrixOp
    using RecursiveArrayTools: ArrayPartition
    using Random

    Random.seed!(0)

    A1 = T[  1.0  -2.0   3.0;
             2.0  -1.0   0.0;
            -1.0   0.0   4.0;
            -1.0  -1.0  -1.0]
    A2 = T[ -4.0  5.0;
            -1.0  3.0;
            -3.0  2.0;
             1.0  3.0]
    A = hcat(MatrixOp(A1), MatrixOp(A2))
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n1 = size(A1)
    _, n2 = size(A2)
    n = n1 + n2

    R = real(T)

    lam = R(0.1)*norm(A'*b, Inf)
    @test typeof(lam) == R

    f = Translate(SqrNormL2(R(1)), -b)
    g = SeparableSum(NormL1(lam), NormL1(lam))

    x_star = ArrayPartition(
        T[-3.877278911564627e-01, 0, 0],
        T[2.174149659863943e-02, 6.168435374149660e-01]
    )

    TOL = 1e-4

    @testset "FBS" begin

        ## Nonfast/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.forwardbackward(x0, f=f, A=A, g=g, L=opnorm([A1 A2])^2, tol=TOL)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150

        # Nonfast/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.forwardbackward(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300

        # Fast/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.forwardbackward(x0, f=f, A=A, g=g, L=opnorm([A1 A2])^2, tol=TOL, fast=true)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100

        # Fast/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.forwardbackward(x0, f=f, A=A, g=g, adaptive=true, tol=TOL, fast=true)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
    end

    @testset "ZeroFPR" begin

        # ZeroFPR/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.zerofpr(x0, f=f, A=A, g=g, L=opnorm([A1 A2])^2, tol=TOL)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        # ZeroFPR/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.zerofpr(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "PANOC" begin

        # PANOC/Nonadaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.panoc(x0, f=f, A=A, g=g, L=opnorm([A1 A2])^2, tol=TOL)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        ## PANOC/Adaptive

        x0 = ArrayPartition(zeros(T, n1), zeros(T, n2))
        x, it = ProximalAlgorithms.panoc(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

end
