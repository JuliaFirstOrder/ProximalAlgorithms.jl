using Zygote
using AbstractDifferentiation: ZygoteBackend
using ProximalOperators: LeastSquares, NormL1
using ProximalAlgorithms
using LinearAlgebra

@testset "Verbose" for T in [Float64]
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

    f_autodiff = ProximalAlgorithms.AutoDifferentiable(x -> (norm(x - b)^2)/2, ZygoteBackend())
    fA_autodiff = ProximalAlgorithms.AutoDifferentiable(x -> (norm(A*x - b)^2)/2, ZygoteBackend())
    fA_prox = LeastSquares(A, b)
    g = NormL1(lam)

    Lf = opnorm(A)^2

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ForwardBackward" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150

        # Nonfast/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ForwardBackward(
            tol = TOL,
            adaptive = true,
            verbose = true,
        )
        x, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300

        # Fast/Nonadaptive

        x0 = zeros(T, n)
        solver =
            ProximalAlgorithms.FastForwardBackward(tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100

        # Fast/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.FastForwardBackward(
            tol = TOL,
            adaptive = true,
            verbose = true,
        )
        x, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
    end

    @testset "ZeroFPR" begin

        # ZeroFPR/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ZeroFPR(tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        # ZeroFPR/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "PANOC" begin

        # PANOC/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.PANOC(tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        ## PANOC/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "PANOCplus" begin

        # PANOCplus/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.PANOCplus(tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g, Lf = Lf)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        ## PANOCplus/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL, verbose = true)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "DouglasRachford" begin

        # Douglas-Rachford

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.DouglasRachford(
            gamma = R(10) / Lf,
            tol = TOL,
            verbose = true,
        )
        y, it = solver(x0 = x0, f = fA_prox, g = g)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 30

    end

end
