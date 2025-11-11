using LinearAlgebra
using ProximalOperators: NormL1, SqrNormL2, ElasticNet, Translate
using ProximalAlgorithms
using Zygote
using DifferentiationInterface: AutoZygote

@testset "Elastic net ($T)" for T in [Float32, Float64, ComplexF32, ComplexF64]
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    reg = ElasticNet(R(1), R(1))
    reg1 = NormL1(R(1))
    reg2 = SqrNormL2(R(1))
    loss = Translate(SqrNormL2(R(1)), -b)
    cost =
        ProximalAlgorithms.AutoDifferentiable(x -> (norm(A * x - b)^2) / 2, AutoZygote())

    L = opnorm(A)^2

    x_star = T[-0.6004983388704322, 0.0, 0.0, 0.195182724252491, 0.764119601328903]

    @testset "DavisYin" begin

        # with known initial iterate

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DavisYin(tol = R(1e-6))
        xf_dys, it_dys = @inferred solver(x0 = x0, f = cost, g = reg1, h = reg2, Lf = L)
        @test eltype(xf_dys) == T
        @test norm(xf_dys - x_star, Inf) <= 1e-3
        @test it_dys <= 140
        @test x0 == x0_backup

        # with random initial iterate

        x0 = randn(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.DavisYin(tol = R(1e-6))
        xf_dys, it_dys = @inferred solver(x0 = x0, f = cost, g = reg1, h = reg2, Lf = L)
        @test eltype(xf_dys) == T
        @test norm(xf_dys - x_star, Inf) <= 1e-3
        @test x0 == x0_backup

    end

    afba_test_params = [(2, 0, 130), (1, 1, 2000), (0, 1, 320), (0, 0, 194), (1, 0, 130)]

    @testset "AFBA" for (theta, mu, maxit) in afba_test_params

        # with known initial iterates

        x0 = zeros(T, n)
        x0_backup = copy(x0)
        y0 = zeros(T, m)
        y0_backup = copy(y0)

        solver = ProximalAlgorithms.AFBA(theta = theta, mu = mu, tol = R(1e-6))
        (x_afba, y_afba), it_afba = solver(
            x0 = x0,
            y0 = y0,
            f = ProximalAlgorithms.AutoDifferentiable(reg2, AutoZygote()),
            g = reg1,
            h = loss,
            L = A,
            beta_f = 1,
        )
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test it_afba <= maxit
        @test x0 == x0_backup
        @test y0 == y0_backup

        # with random initial iterates

        x0 = randn(T, n)
        x0_backup = copy(x0)
        y0 = randn(T, m)
        y0_backup = copy(y0)

        solver = ProximalAlgorithms.AFBA(theta = theta, mu = mu, tol = R(1e-6))
        (x_afba, y_afba), it_afba = solver(
            x0 = x0,
            y0 = y0,
            f = ProximalAlgorithms.AutoDifferentiable(reg2, AutoZygote()),
            g = reg1,
            h = loss,
            L = A,
            beta_f = 1,
        )
        @test eltype(x_afba) == T
        @test eltype(y_afba) == T
        @test norm(x_afba - x_star, Inf) <= 1e-4
        @test x0 == x0_backup
        @test y0 == y0_backup

    end

    @testset "ADMM" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        @testset "$(typeof(ps).name.name)" for ps in [
            ProximalAlgorithms.FixedPenalty(),
            ProximalAlgorithms.ResidualBalancingPenalty(adp_freq = 5),
            # ProximalAlgorithms.WohlbergPenalty(), # TODO: This does not converge, needs parameter tuning
            # ProximalAlgorithms.BarzilaiBorweinPenalty(), # TODO: This does not converge, needs debugging
            ProximalAlgorithms.SpectralRadiusBoundPenalty(),
            ProximalAlgorithms.SpectralRadiusApproximationPenalty(),
        ]
            solver = ProximalAlgorithms.ADMM(tol = R(5e-5), maxit=300, penalty_sequence = ps)
            x_admm, it_admm = @inferred solver(; x0, A, b, g = (reg1, reg2))
            @test eltype(x_admm) == T
            @test norm(x_admm - x_star, Inf) <= 1e-3
            @test it_admm ≤ 300
            @test x0 == x0_backup
        end
    end

end
