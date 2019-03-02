using ProximalOperators
using ProximalAlgorithms
using LinearAlgebra
using Random
using Test

@testset "L1-LogReg small" begin

    A = [  1.0  -2.0   3.0  -4.0  5.0;
           2.0  -1.0   0.0  -1.0  3.0;
          -1.0   0.0   4.0  -3.0  2.0;
          -1.0  -1.0  -1.0   1.0  3.0]
    b = [1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    f = Translate(LogisticLoss(ones(m)), -b)
    lam = 0.1
    g = NormL1(lam)

    x_star = [0, 0, 2.114635341704963e-01, 0, 2.845881348733116e+00]

    TOL = 1e-6

    # Nonfast/Adaptive

    x0 = zeros(n)
    x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, fast=false, adaptive=true, tol=TOL)
    @test norm(x - x_star, Inf) <= 1e-5
    @test it < 1100

    # Fast/Adaptive

    x0 = zeros(n)
    x, it = ProximalAlgorithms.FBS(x0, f=f, A=A, g=g, fast=true, adaptive=true, tol=TOL)
    @test norm(x - x_star, Inf) <= 1e-5
    @test it < 500

    # ZeroFPR/Adaptive

    # x0 = zeros(n)
    # it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fs=f, As=A, g=g, tol=1e-6, adaptive=true)
    # @test norm(x - x_star, Inf) <= 1e-4
    # @test it < 25
    # println(sol)

    # PANOC/Adaptive

    x0 = zeros(n)
    x, it = ProximalAlgorithms.PANOC(x0, f=f, A=A, g=g, adaptive=true, tol=TOL)
    @test norm(x - x_star, Inf) <= 1e-5
    @test it < 50

end
