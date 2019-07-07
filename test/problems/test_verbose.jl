@testset "Verbose" for T in [Float64]

    using ProximalOperators
    using ProximalAlgorithms
    using LinearAlgebra
    using Random

    Random.seed!(0)

    A = T[  1.0  -2.0   3.0  -4.0  5.0;
            2.0  -1.0   0.0  -1.0  3.0;
           -1.0   0.0   4.0  -3.0  2.0;
           -1.0  -1.0  -1.0   1.0  3.0]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    lam = R(0.1)*norm(A'*b, Inf)
    @test typeof(lam) == R

    f = Translate(SqrNormL2(R(1)), -b)
    f2 = LeastSquares(A, b)
    g = NormL1(lam)

    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = R(1e-4)

    @testset "ForwardBackward" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ForwardBackward{R}(tol=TOL, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g, L=opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150

        # Nonfast/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ForwardBackward{R}(tol=TOL, adaptive=true, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300

        # Fast/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ForwardBackward{R}(tol=TOL, fast=true, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g, L=opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100

        # Fast/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ForwardBackward{R}(tol=TOL, adaptive=true, fast=true, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
    end

    @testset "ZeroFPR" begin

        # ZeroFPR/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ZeroFPR{R}(tol=TOL, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g, L=opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        # ZeroFPR/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.ZeroFPR{R}(adaptive=true, tol=TOL, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "PANOC" begin

        # PANOC/Nonadaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.PANOC{R}(tol=TOL, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g, L=opnorm(A)^2)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        ## PANOC/Adaptive

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.PANOC{R}(adaptive=true, tol=TOL, verbose=true)
        x, it = solver(x0, f=f, A=A, g=g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "DouglasRachford" begin

        # Douglas-Rachford

        x0 = zeros(T, n)
        solver = ProximalAlgorithms.DouglasRachford{R}(gamma=R(10.0)/opnorm(A)^2, tol=TOL, verbose=true)
        y, z, it = solver(x0, f=f2, g=g)
        @test eltype(y) == T
        @test eltype(z) == T
        @test norm(y - x_star, Inf) <= TOL
        @test norm(z - x_star, Inf) <= TOL
        @test it < 30

    end

end
