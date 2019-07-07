@testset "Linear programs ($T)" for T in [Float32, Float64]

    using ProximalOperators
    using ProximalAlgorithms
    using LinearAlgebra
    using Random

    Random.seed!(0)

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

    function assert_lp_solution(c, A, b, x, y, tol)
        # Check and print solution quality measures (for some reason the
        # returned dual iterate is the negative of the dual LP variable y)

        nonneg = -minimum(min.(0.0, x))
        @test nonneg <= tol

        primal_feasibility = norm(A*x - b)
        @test primal_feasibility <= tol

        dual_feasibility = maximum(max.(0.0, -A'*y - c))
        @test dual_feasibility <= tol

        complementarity = abs(dot(c + A'*y, x))
        @test complementarity <= tol
    end

    n = 100 # primal dimension
    m = 80 # dual dimension (i.e. number of linear equalities)
    k = 50 # number of active dual constraints (must be 0 <= k <= n)
    x_star = vcat(rand(T, k), zeros(T, n-k)) # primal optimal point
    s_star = vcat(zeros(T, k), rand(T, n-k)) # dual optimal slack variable
    y_star = randn(T, m) # dual optimal point

    A = randn(T, m, n)
    b = A*x_star
    c = A'*y_star + s_star

    tol = 100*eps(T)
    maxit = 10_000

    @testset "AFBA" begin

        f = Linear(c)
        g = IndNonnegative()
        h = IndPoint(b)

        x0 = zeros(T, n)
        y0 = zeros(T, m)

        solver = ProximalAlgorithms.AFBA{T}(tol=tol, maxit=maxit)
        x, y, it = solver(x0, y0, f=f, g=g, h=h, L=A)

        @test eltype(x) == T
        @test eltype(y) == T

        @test it <= maxit

        assert_lp_solution(c, A, b, x, y, 1e2*tol)

    end

    @testset "VuCondat" begin

        f = Linear(c)
        g = IndNonnegative()
        h = IndPoint(b)

        x0 = zeros(T, n)
        y0 = zeros(T, m)

        solver = ProximalAlgorithms.VuCondat(T, tol=tol, maxit=maxit)
        x, y, it = solver(x0, y0, f=f, g=g, h=h, L=A)

        @test eltype(x) == T
        @test eltype(y) == T

        @test it <= maxit

        assert_lp_solution(c, A, b, x, y, 1e2*tol)

    end

    # TODO: add Chambolle-Pock (using separable sum of IndPoint and IndNonnegative)

    @testset "DavisYin" begin

        f = IndAffine(A, b)
        g = IndNonnegative()
        h = Linear(c)

        x0 = zeros(T, n)

        solver = ProximalAlgorithms.DavisYin{T}(gamma=T(1), tol=tol, maxit=maxit)
        xf, xg, it = solver(x0, f=f, g=g, h=h)

        @test eltype(xf) == T
        @test eltype(xg) == T

        @test it <= maxit

        @assert norm(xf - x_star) <= 1e2*tol
        @assert norm(xg - x_star) <= 1e2*tol

    end

end
