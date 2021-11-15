@testset "Sparse logistic small ($T)" for T in [Float32, Float64]
    using ProximalOperators
    using ProximalAlgorithms
    using LinearAlgebra

    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    f = Translate(LogisticLoss(ones(R, m), R(1)), -b)
    f2 = ComposeAffine(LogisticLoss(ones(R, m), R(1)), A, -b)
    lam = R(0.1)
    g = NormL1(lam)

    x_star = T[0, 0, 2.114635341704963e-01, 0, 2.845881348733116e+00]

    TOL = R(1e-6)

    # Nonfast/Adaptive

    x0 = zeros(T, n)
    x0_backup = copy(x0)
    solver = ProximalAlgorithms.ForwardBackward(tol = TOL, adaptive = true)
    x, it = solver(x0, f = f2, g = g)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 1100
    @test x0 == x0_backup

    # Fast/Adaptive

    x0 = zeros(T, n)
    x0_backup = copy(x0)
    solver = ProximalAlgorithms.FastForwardBackward(tol = TOL, adaptive = true)
    x, it = solver(x0, f = f2, g = g)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 500
    @test x0 == x0_backup

    # ZeroFPR/Adaptive

    x0 = zeros(T, n)
    x0_backup = copy(x0)
    solver = ProximalAlgorithms.ZeroFPR(adaptive = true, tol = TOL)
    x, it = solver(x0, f = f, A = A, g = g)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 25
    @test x0 == x0_backup

    # PANOC/Adaptive

    x0 = zeros(T, n)
    x0_backup = copy(x0)
    solver = ProximalAlgorithms.PANOC(adaptive = true, tol = TOL)
    x, it = solver(x0, f = f, A = A, g = g)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 50
    @test x0 == x0_backup

    # NOLIP/Adaptive

    x0 = zeros(T, n)
    x0_backup = copy(x0)
    solver = ProximalAlgorithms.NOLIP(adaptive = true, tol = TOL)
    x, it = solver(x0, f = f, A = A, g = g)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 50
    @test x0 == x0_backup

end
