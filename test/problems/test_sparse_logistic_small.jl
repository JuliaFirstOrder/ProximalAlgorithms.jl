@testset "Sparse logistic small ($T)" for T in [Float32, Float64]

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

    f = Translate(LogisticLoss(ones(R, m), R(1)), -b)
    lam = R(0.1)
    g = NormL1(lam)

    x_star = T[0, 0, 2.114635341704963e-01, 0, 2.845881348733116e+00]

    TOL = 1e-6

    # Nonfast/Adaptive

    x0 = zeros(T, n)
    x, it = ProximalAlgorithms.forwardbackward(x0, f=f, A=A, g=g, fast=false, adaptive=true, tol=TOL)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 1100

    # Fast/Adaptive

    x0 = zeros(T, n)
    x, it = ProximalAlgorithms.forwardbackward(x0, f=f, A=A, g=g, fast=true, adaptive=true, tol=TOL)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 500

    # ZeroFPR/Adaptive

    x0 = zeros(T, n)
    x, it = ProximalAlgorithms.zerofpr(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 25

    # PANOC/Adaptive

    x0 = zeros(T, n)
    x, it = ProximalAlgorithms.panoc(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= 1e-4
    @test it < 50

end
