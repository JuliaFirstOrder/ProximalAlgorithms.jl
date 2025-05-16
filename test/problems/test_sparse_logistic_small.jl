using Zygote
using DifferentiationInterface: AutoZygote
using ProximalOperators: NormL1
using ProximalAlgorithms
using LinearAlgebra

@testset "Sparse logistic small ($T)" for T in [Float32, Float64]
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    function logistic_loss(logits)
        u = 1 .+ exp.(-logits)  # labels are assumed all one
        return sum(log.(u))
    end

    f_autodiff =
        ProximalAlgorithms.AutoDifferentiable(x -> logistic_loss(x - b), AutoZygote())
    fA_autodiff = ProximalAlgorithms.AutoDifferentiable(
        x -> logistic_loss(A * x - b),
        AutoZygote(),
    )
    lam = R(0.1)
    g = NormL1(lam)

    x_star = T[0, 0, 2.114635341704963e-01, 0, 2.845881348733116e+00]

    TOL = R(1e-6)

    @testset "ForwardBackward (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 1100
        @test x0 == x0_backup
    end

    @testset "ForwardBackward (adaptive step, regret)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ForwardBackward(
            tol = TOL,
            adaptive = true,
            increase_gamma = R(1.01),
        )
        x, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 500
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
        x, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 500
        @test x0 == x0_backup
    end

    @testset "FastForwardBackward (adaptive step, regret)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.FastForwardBackward(
            tol = TOL,
            adaptive = true,
            increase_gamma = R(1.01),
        )
        x, it = solver(x0 = x0, f = fA_autodiff, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 200
        @test x0 == x0_backup
    end

    @testset "ZeroFPR (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 25
        @test x0 == x0_backup
    end

    @testset "ZeroFPR (adaptive, nonmonotone)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL, monotonicity=R(0.5))
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 30
        @test x0 == x0_backup
    end

    @testset "PANOC (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 50
        @test x0 == x0_backup
    end

    @testset "PANOCplus (adaptive step)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL)
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 50
        @test x0 == x0_backup
    end

    @testset "PANOCplus (adaptive step, nonmonotone)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL, monotonicity=R(0.9))
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 50
        @test x0 == x0_backup
    end

    @testset "PANOCplus (adaptive step, very nonmonotone)" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus(adaptive = true, tol = TOL, monotonicity=R(0.1))
        x, it = solver(x0 = x0, f = f_autodiff, A = A, g = g)
        @test eltype(x) == T
        @test norm(x - x_star, Inf) <= 1e-4
        @test it < 110
        @test x0 == x0_backup
    end

end
