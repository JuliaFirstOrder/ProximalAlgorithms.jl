using LinearAlgebra
using Test

using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators: NormL1, LeastSquares
using ProximalAlgorithms

@testset "Lasso small (strongly convex, $T)" for T in [Float32, Float64]
    dim = 5
    mf = T(1)
    Lf = T(10)

    x_star = T[
        0.8466800540711814,
        0.17674262101590932,
        -0.4987234606672925,
        0.5531315167924573,
        -0.14739365562631113,
    ]

    lam = (mf + Lf) / 2

    w = T[
        0.15823052457732423,
        0.6874613398393697,
        0.9357764685973888,
        0.05863707298785681,
        0.49087050154723844,
    ]
    D = Diagonal(sqrt(mf) .+ (sqrt(Lf) - sqrt(mf)) * w)
    D[1] = sqrt(mf)
    D[end] = sqrt(Lf)

    B = T[
        0.6997086717991196 0.37124544422925876 0.31840520080247225 0.20097960566711592 0.038329117953706526
        0.1134636504826555 0.8273912343075426 0.8997522727456534 0.9821118072706589 0.9100659142463259
        0.9701886480567284 0.42825250593295605 0.6952640061565183 0.9699979632534245 0.6106722979088736
        0.4442755181780246 0.4641748710746476 0.9716060376558348 0.5951146731055232 0.5699044913634803
        0.6681510415197733 0.35423403325449887 0.28461925562068024 0.15941152427241456 0.6499046326711716
    ]
    Q = qr(B).Q
    A = Q * D * Q'
    b = A * x_star + lam * inv(A') * sign.(x_star)

    fA_prox = LeastSquares(A, b)
    fA_autodiff =
        ProximalAlgorithms.AutoDifferentiable(x -> (norm(A * x - b)^2) / 2, AutoZygote())
    g = NormL1(lam)

    TOL = T(1e-4)

    x0 = A \ b
    x0_backup = copy(x0)

    @testset "SFISTA" begin
        solver = ProximalAlgorithms.SFISTA(tol = TOL)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf, mf = mf)
        @test eltype(y) == T
        @test norm(y - x_star) <= TOL
        @test it < 40
        @test x0 == x0_backup
    end

    @testset "ForwardBackward" begin
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 110
        @test x0 == x0_backup
    end

    @testset "ForwardBackward (adaptive step)" begin
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 300
        @test x0 == x0_backup
    end

    @testset "ForwardBackward (adaptive step, regret)" begin
        solver = ProximalAlgorithms.ForwardBackward(
            tol = TOL,
            adaptive = true,
            increase_gamma = T(1.01),
        )
        y, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 80
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward" begin
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf, mf = mf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 35
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step)" begin
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step, regret)" begin
        solver = ProximalAlgorithms.FastForwardBackward(
            tol = TOL,
            adaptive = true,
            increase_gamma = T(1.01),
        )
        y, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 100
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (custom extrapolation)" begin
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
        y, it = solver(
            x0 = x0,
            f = fA_autodiff,
            g = g,
            gamma = 1 / Lf,
            mf = mf,
            extrapolation_sequence = ProximalAlgorithms.ConstantNesterovSequence(
                mf,
                1 / Lf,
            ),
        )
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 35
        @test x0 == x0_backup
    end

    @testset "DRLS" begin
        solver = ProximalAlgorithms.DRLS(tol = TOL)
        v, it = solver(x0 = x0, f = fA_prox, g = g, mf = mf)
        @test eltype(v) == T
        @test norm(v - x_star, Inf) <= TOL
        @test it < 14
        @test x0 == x0_backup
    end

    @testset "PANOC" begin
        solver = ProximalAlgorithms.PANOC(tol = TOL)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 45
        @test x0 == x0_backup
    end

    @testset "PANOCplus" begin
        solver = ProximalAlgorithms.PANOCplus(tol = TOL)
        y, it = solver(x0 = x0, f = fA_autodiff, g = g, Lf = Lf)
        @test eltype(y) == T
        @test norm(y - x_star, Inf) <= TOL
        @test it < 45
        @test x0 == x0_backup
    end

    @testset "ADMM" begin
        @testset "$(typeof(ps).name.name)" for ps in [
            ProximalAlgorithms.FixedPenalty(),
            ProximalAlgorithms.ResidualBalancingPenalty(),
            ProximalAlgorithms.WohlbergPenalty(),
            # ProximalAlgorithms.BarzilaiBorweinSpectralPenalty(), # TODO: This does not converge, needs debugging
            # ProximalAlgorithms.SpectralRadiusBoundPenalty(), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.SpectralRadiusApproximationPenalty(), # TODO: This does not converge, needs parameter tuning
        ]
            solver = ProximalAlgorithms.ADMM(tol = 1e-5, maxit=1000, penalty_sequence = ps)
            x_admm, it_admm = @inferred solver(; x0, A, b, g)
            @test eltype(x_admm) == T
            @test norm(x_admm - x_star, Inf) <= 1e-3
            @test it_admm < 50
            @test x0 == x0_backup
        end
    end

end
