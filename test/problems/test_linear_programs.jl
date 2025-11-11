using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators: Linear, IndNonnegative, IndPoint, IndAffine, SlicedSeparableSum
using ProximalAlgorithms
using LinearAlgebra

function assert_lp_solution(c, A, b, x, y, tol)
    # Check and print solution quality measures (for some reason the
    # returned dual iterate is the negative of the dual LP variable y)

    nonneg = -minimum(min.(0.0, x))
    @test nonneg <= tol

    primal_feasibility = norm(A * x - b)
    @test primal_feasibility <= tol

    dual_feasibility = maximum(max.(0.0, -A' * y - c))
    @test dual_feasibility <= tol

    complementarity = abs(dot(c + A' * y, x))
    @test complementarity <= tol
end

@testset "Linear programs ($T)" for T in [Float32, Float64]

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

    n = 10 # primal dimension
    m = 8 # dual dimension (i.e. number of linear equalities)
    k = 5 # number of active dual constraints (must be 0 <= k <= n)
    # primal optimal point
    x_star = T[
        0.03606099647643202,
        0.6641306619990367,
        0.14689326835593086,
        0.8616058527226432,
        0.6518888386753204,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    # dual optimal slack variable
    s_star = T[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.7652088547602296,
        0.5474850582278736,
        0.6291635905927829,
        0.5505791259743739,
        0.2787959059899776,
    ]
    # dual optimal point
    y_star = T[
        -0.18584225497010673,
        1.0289179383971105,
        -0.5464276767930338,
        0.6916826379378143,
        0.0052105118567898154,
        -1.3874339670318891,
        2.6448648322602337,
        0.4436510755789434,
    ]
    # equality constraints coefficients
    A = T[
        0.9670566934980286 0.3533472582831891 -0.2819205403663176 -0.3071751292615649 -0.008197151577028876 1.9968628487924958 -1.3978028828102382 -0.10109443461339453 0.3540170332321368 -0.5983697948494805
        -0.08643002247762153 1.0822796020885652 -0.8133412010128359 0.5073579509595517 -0.2793798574957181 -0.3437059320143032 -0.10596751644319548 0.8296541102523696 -0.28406001552384064 0.33566140783820164
        -0.6474810505335304 1.2524424221527595 0.7261766049639965 0.6930548839308885 0.9064585656428545 0.5197152355759463 1.3746077918877961 1.5289983684262054 0.9258506062644877 -1.3774011601531342
        0.8725430405646543 1.1911309824177332 -0.7221185305116811 0.44324697589700257 -0.15551892138880116 -0.5167033349322372 -1.4774045165687548 -0.8670756465083638 -1.4024841888738206 0.296570674868075
        0.5252662593900623 -0.6817132148621747 0.37446015899900237 0.4180282147408251 -0.8405436435394317 -1.742941478155391 -0.06419427972299957 0.43910205564784205 1.0643883425210827 2.3063869854427335
        -0.7148913270640012 -0.769028546306448 -1.059257097999333 -0.6795170119545777 -0.0498976886779524 -1.2392107698826862 -0.4415384005606088 -1.058758868936871 -0.108504245219676 -0.5576550366602419
        -1.208388774142618 -0.15206129387542855 2.311520055340236 0.8043266793420988 -0.5692874893454578 1.1246423711381501 0.5335942753441769 2.6595405998250876 0.09162292399585106 0.3749905036072034
        -0.2300660921924555 -0.7014271654627467 -0.20170532145095504 -0.02503019691724233 -2.2191605023268512 0.9110653907470295 1.8238644805628141 -1.1908921287611471 0.12168786553115268 0.17399181994853638
    ]
    b = A * x_star
    c = A' * y_star + s_star

    tol = 100 * eps(T)
    maxit = 100_000

    @testset "AFBA" begin

        f = ProximalAlgorithms.AutoDifferentiable(x -> dot(c, x), AutoZygote())
        g = IndNonnegative()
        h = IndPoint(b)

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        y0 = zeros(T, m)
        y0_backup = copy(y0)

        solver = ProximalAlgorithms.AFBA(tol = tol, maxit = maxit)
        (x, y), it = solver(x0 = x0, y0 = y0, f = f, g = g, h = h, L = A, beta_f = 0)

        @test eltype(x) == T
        @test eltype(y) == T

        @test it <= maxit

        assert_lp_solution(c, A, b, x, y, 1000 * tol)

        @test x0 == x0_backup
        @test y0 == y0_backup

    end

    @testset "VuCondat" begin

        f = ProximalAlgorithms.AutoDifferentiable(x -> dot(c, x), AutoZygote())
        g = IndNonnegative()
        h = IndPoint(b)

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        y0 = zeros(T, m)
        y0_backup = copy(y0)

        solver = ProximalAlgorithms.VuCondat(tol = tol, maxit = maxit)
        (x, y), it = solver(x0 = x0, y0 = y0, f = f, g = g, h = h, L = A, beta_f = 0)

        @test eltype(x) == T
        @test eltype(y) == T

        @test it <= maxit

        assert_lp_solution(c, A, b, x, y, 1000 * tol)

        @test x0 == x0_backup
        @test y0 == y0_backup

    end

    @testset "ChambollePock" begin
        g = Linear(c)
        h = SlicedSeparableSum((IndPoint(b), IndNonnegative()), ((1:m,), (m+1:m+n,)))

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        y0 = zeros(T, m + n)
        y0_backup = copy(y0)

        solver = ProximalAlgorithms.ChambollePock(tol = tol, maxit = maxit)
        (x, y), it = solver(x0 = x0, y0 = y0, g = g, h = h, L = vcat(A, Matrix{T}(I, n, n)))

        @test eltype(x) == T
        @test eltype(y) == T

        @test it <= maxit

        assert_lp_solution(c, A, b, x, y[1:m], 1000 * tol)

        @test x0 == x0_backup
        @test y0 == y0_backup
    end

    @testset "DavisYin" begin
        f = ProximalAlgorithms.AutoDifferentiable(x -> dot(c, x), AutoZygote())
        g = IndNonnegative()
        h = IndAffine(A, b)

        x0 = zeros(T, n)
        x0_backup = copy(x0)

        solver = ProximalAlgorithms.DavisYin(gamma = T(1), tol = tol, maxit = maxit)
        xf, it = solver(x0 = x0, f = f, g = g, h = h)

        @test eltype(xf) == T
        @test it <= maxit
        @test norm(xf - x_star) <= 1e2 * tol
        @test x0 == x0_backup
    end

    @testset "ADMM" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        @testset "$(typeof(ps).name.name)" for ps in [
            ProximalAlgorithms.FixedPenalty(),
            # ProximalAlgorithms.ResidualBalancingPenalty(normalized=true), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.WohlbergPenalty(), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.BarzilaiBorweinSpectralPenalty(), # TODO: This does not converge, needs debugging
            # ProximalAlgorithms.SpectralRadiusBoundPenalty(), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.SpectralRadiusApproximationPenalty(), # TODO: This does not converge, needs parameter tuning
        ]
            solver = ProximalAlgorithms.ADMM(; tol=1e-6, maxit=10000, penalty_sequence = ps)
            x_admm, it_admm = @inferred solver(; x0, g = (Linear(c), IndNonnegative(), IndPoint(b)), B = (I, I, A))
            @test eltype(x_admm) == T
            @test norm(x_admm - x_star, Inf) <= 1e-3
            @test it_admm < maxit
            @test x0 == x0_backup
        end
    end

end
