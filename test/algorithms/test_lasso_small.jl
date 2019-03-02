using ProximalOperators
using ProximalAlgorithms
using LinearAlgebra
using Random
using Test

@testset "Lasso small" begin

    A = [  1.0  -2.0   3.0  -4.0  5.0;
           2.0  -1.0   0.0  -1.0  3.0;
          -1.0   0.0   4.0  -3.0  2.0;
          -1.0  -1.0  -1.0   1.0  3.0]
    b = [1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    f = Translate(SqrNormL2(), -b)
    f2 = LeastSquares(A, b)
    lam = 0.1*norm(A'*b, Inf)
    g = NormL1(lam)

    x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

    TOL = 1e-4

    @testset "FBS" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, L=opnorm(A)^2, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 150

        # # testing solver already at solution
        # it, x = ProximalAlgorithms.run!(sol)
        # @test it == 1

        # Nonfast/Adaptive

        x0 = zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 300

        # Fast/Nonadaptive

        x0 = zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, L=opnorm(A)^2, tol=TOL, fast=true)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 100

        # testing solver already at solution
        # it, x = ProximalAlgorithms.run!(sol)
        # @test it == 1

        # Fast/Adaptive

        x0 = zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, adaptive=true, tol=TOL, fast=true)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200
    end

    @testset "ZeroFPR" begin

        # # ZeroFPR/Nonadaptive
        #
        # x0 = zeros(n)
        # it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
        # @test norm(x - x_star, Inf) <= TOL
        # @test it < 15
        # println(sol)
        #
        # #testing solver already at solution
        # it, x = ProximalAlgorithms.run!(sol)
        # @test it == 1
        #
        # # ZeroFPR/Adaptive
        #
        # x0 = zeros(n)
        # it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, adaptive=true)
        # @test norm(x - x_star, Inf) <= TOL
        # @test it < 15

    end

    @testset "PANOC" begin

        # PANOC/Nonadaptive

        x0 = zeros(n)
        x, it = ProximalAlgorithms.PANOC(x0, f=f, A=A, g=g, L=opnorm(A)^2, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

        ## PANOC/Adaptive

        x0 = zeros(n)
        x, it = ProximalAlgorithms.PANOC(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    @testset "DRS" begin

        # Douglas-Rachford

        x0 = zeros(n)
        y, z, it = ProximalAlgorithms.DRS(x0, f=f2, g=g, gamma=10.0/opnorm(A)^2, tol=TOL)
        @test norm(y - x_star, Inf) <= TOL
        @test norm(z - x_star, Inf) <= TOL
        @test it < 30

    end

end

@testset "Lasso complex" begin

    #####################
    # Complex Variables #
    #####################

    Random.seed!(123)
    m, n = 10,5

    A = randn(m,n)+im*randn(m,n)
    b = randn(m)+im*randn(m)

    f = Translate(SqrNormL2(), -b)
    f2 = LeastSquares(A, b)
    lam = 0.01*norm(A'*b, Inf)
    g = NormL1(lam)

    x0 = zeros(n)+im*zeros(n)
    x_star, it = ProximalAlgorithms.PANOC(x0, f=f, A=A, g=g, L=opnorm(A)^2, verbose=false)

    TOL = 1e-4

    @testset "FBS" begin

        ## Nonfast/Nonadaptive

        x0 = zeros(n)+im*zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, L=opnorm(A)^2, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 200

        # # testing solver already at solution
        # it, x = ProximalAlgorithms.run!(sol)
        # @test it == 1

        # Nonfast/Adaptive

        x0 = zeros(n)+im*zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 250

        # Fast/Nonadaptive

        x0 = zeros(n)+im*zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, L=opnorm(A)^2, tol=TOL, fast=true)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 120

        # # testing solver already at solution
        # it, x = ProximalAlgorithms.run!(sol)
        # @test it == 1

        # Fast/Adaptive

        x0 = zeros(n)+im*zeros(n)
        x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, adaptive=true, tol=TOL, fast=true)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 120

    end

    @testset "ZeroFPR" begin

        # # ZeroFPR/Nonadaptive
        #
        # x0 = zeros(n)+im*zeros(n)
        # it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
        # @test norm(x - x_star, Inf) <= TOL
        # @test it < 15
        # println(sol)
        #
        # #testing solver already at solution
        # it, x = ProximalAlgorithms.run!(sol)
        # @test it == 1
        #
        # # ZeroFPR/Adaptive
        #
        # x0 = zeros(n)+im*zeros(n)
        # it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, adaptive=true)
        # @test norm(x - x_star, Inf) <= TOL
        # @test it < 15

    end

    @testset "PANOC" begin

        # PANOC/Nonadaptive

        x0 = zeros(n)+im*zeros(n)
        x, it = ProximalAlgorithms.PANOC(x0, f=f, A=A, g=g, L=opnorm(A)^2, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 25

        ## PANOC/Adaptive

        x0 = zeros(n)+im*zeros(n)
        x, it = ProximalAlgorithms.PANOC(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
        @test norm(x - x_star, Inf) <= TOL
        @test it < 20

    end

    # #############################################
    # # Real Variables  mapped to Complex Numbers #
    # #############################################
    #
    # Random.seed!(123)
    # n = 2^6
    # A = AbstractOperators.DFT(n)[1:div(n,2)]      # overcomplete dictionary
    #
    # x = sprandn(n,0.5)
    # b = fft(x)[1:div(n,2)]
    #
    # #f = Translate(LogisticLoss(ones(n)), -b)
    # f = Translate(SqrNormL2(), -b)
    # lam = 0.01*norm(A'*b,Inf)
    # g = NormL1(lam)
    #
    # x0 = zeros(n)
    # it, x_star, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, verbose = 0, tol = 1e-8)
    #
    # # Nonfast/Adaptive
    #
    # x0 = zeros(n)
    # it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true)
    # @test norm(x - x_star, Inf) <= TOL
    # @test it < 50
    # println(sol)
    #
    # # Fast/Adaptive
    #
    # x0 = zeros(n)
    # it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true, fast=true)
    # @test norm(x - x_star, Inf) <= TOL
    # @test it < 50
    # println(sol)
    #
    # # ZeroFPR/Adaptive
    #
    # x0 = zeros(n)
    # it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true)
    # @test norm(x - x_star, Inf) <= TOL
    # @test it < 15
    # println(sol)
    #
    # # PANOC/Adaptive
    #
    # x0 = zeros(n)
    # it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true)
    # @test norm(x - x_star, Inf) <= TOL
    # @test it < 20
    # println(sol)

end
