using ProximalOperators
using ProximalAlgorithms
using LinearAlgebra
using Random
using Test

Random.seed!(0)

@testset "Primal-dual" begin

    ### this test includes two tests

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

    tol = 1e-5

    test_params = [
        Dict(   "theta"      => 2.0,
                "mu"         => 0.0,
                "it"         => (70,135,150),
        ),
        Dict(   "theta"      => 1.0,
                "mu"         => 1.0,
                "it"         => (80,130,150),
        ),
        Dict(   "theta"      => 0.0,
                "mu"         => 1.0,
                "it"         => (90,345,200),
        ),
        Dict(   "theta"      => 0.0,
                "mu"         => 0.0,
                "it"         => (80,170,250),
        ),
        Dict(   "theta"      => 1.0,
                "mu"         => 0.0,
                "it"         => (90,130,350),
        )
    ]

    @testset "Lasso small" begin

        ## 1-testing combinations with two terms: Lasso

        x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

        for i = 1:length(test_params)

            theta = test_params[i]["theta"]
            mu    = test_params[i]["mu"]
            itnum = test_params[i]["it"]

            x0 = randn(n)

            # h\equiv 0 (FBS)
            y0 = randn(n)
            x, y, it = ProximalAlgorithms.AFBA(x0, y0; g=g, f=f2, betaQ=opnorm(A'*A), theta=theta, mu=mu, tol=tol)
            @test norm(x - x_star, Inf) <= 1e-4
            @test it <= itnum[1]

            # f=\equiv 0 (Chambolle-Pock)
            y0 = randn(m)
            x, y, it = ProximalAlgorithms.AFBA(x0, y0; g=g, h=f, L=A, theta=theta, mu=mu, tol=tol)
            @test norm(x - x_star, Inf) <= 1e-4
            @test it <= itnum[2]

            # g\equiv 0
            y0 = randn(n) # since L= Identity
            x, y, it = ProximalAlgorithms.AFBA(x0, y0; h=g, f=f2, betaQ=opnorm(A'*A), theta=theta, mu=mu, tol=tol)
            @test norm(x - x_star, Inf) <= 1e-4
            @test it <= itnum[3]

        end

    end

    @testset "ElasticNet" begin

        ## 2- testing with three terms: 1/2\|Ax-b\|^2+ λ\|x\|_1 + + λ_2*\|x\|^2

        lam2 = 1.0

        x0 = randn(n)
        y0 = randn(n)
        itnum = ((130,130),(150,150),(150,150),(200,200),(250,250)); # the number of iterations

        for i = 1:length(test_params)

            theta = test_params[i]["theta"]
            mu    = test_params[i]["mu"]

            x, y, it = ProximalAlgorithms.AFBA(x0, y0; g=g, f=f2, h=SqrNormL2(lam2), betaQ=opnorm(A'*A), theta=theta, mu=mu, tol=tol)

             # the optimality conditions
            temp = lam*sign.(x) + A'*(A*x-b) + lam2*x
            ind= findall(x -> abs.(x)<1e-8,x)
            t1= length(findall((temp[ind] .<= lam .* temp[ind] .>=-lam)==false))
            t2= length(findall((abs.(deleteat!(temp,ind)) .<1e-8)==false))
            @test t1+t2 == 0
            @test it <= itnum[i][1]

            x, y, it = ProximalAlgorithms.AFBA(x0, y0; h=g, f=f2, g=SqrNormL2(lam2), betaQ=opnorm(A'*A), theta=theta, mu=mu, tol=tol)

             # the optimality conditions
            temp = lam*sign.(x) + A'*(A*x-b) + lam2*x
            ind= findall(x -> abs.(x)<1e-8,x)
            t1= length(findall((temp[ind] .<= lam .* temp[ind] .>=-lam)==false))
            t2= length(findall((abs.(deleteat!(temp,ind)) .<1e-8)==false))
            @test t1+t2 == 0
            @test it <= itnum[i][2]

        end

    end

    @testset "LP" begin

        # Solving LP with AFBA
        #
        #   minimize    c'x             -> f = <c,.>
        #   subject to  Ax = b          -> h = ind{b}, L = A
        #               x >= 0          -> g = ind{>=0}
        #
        # Dual LP
        #
        #   maximize    b'y
        #   subject to  A'y <= c
        #
        # Optimality conditions
        #
        #   x >= 0              [primal feasibility 1]
        #   Ax = b              [primal feasibility 2]
        #   A'y <= c            [dual feasibility]
        #   X'(c - A'y) = 0     [complementarity slackness]
        #

        n = 100 # primal dimension
        m = 80 # dual dimension (i.e. number of linear equalities)
        k = 50 # number of active dual constraints (must be 0 <= k <= n)
        x_star = vcat(rand(k), zeros(n-k)) # primal optimal point
        s_star = vcat(zeros(k), rand(n-k)) # dual optimal slack variable
        y_star = randn(m) # dual optimal point

        A = randn(m, n)
        b = A*x_star
        c = A'*y_star + s_star

        f = Linear(c)
        g = IndNonnegative()
        h = IndPoint(b)

        x0 = zeros(n)
        y0 = zeros(m)

        x, y, it = ProximalAlgorithms.AFBA(x0, y0; f=f, g=g, h=h, L=A, tol=1e-8, maxit=10000)

        # Check and print solution quality measures
        # (for some reason the returned dual iterate is the negative of the dual LP variable y)

        TOL_ASSERT = 1e-6

        nonneg = -minimum(min.(0.0, x))
        # println("Nonnegativity          : ", nonneg)
        @test nonneg <= TOL_ASSERT

        primal_feasibility = norm(A*x - b)
        # println("Primal feasibility     : ", primal_feasibility)
        @test primal_feasibility <= TOL_ASSERT

        dual_feasibility = maximum(max.(0.0, -A'*y - c))
        # println("Dual feasibility       : ", dual_feasibility)
        @test dual_feasibility <= TOL_ASSERT

        complementarity = abs(dot(c + A'*y, x))
        # println("Complementarity        : ", complementarity)
        @test complementarity <= TOL_ASSERT

        # println("Primal objective       : ", dot(c, x))
        # println("Dual objective         : ", dot(b, -y))

    end

end

# TODO: add test including function l
