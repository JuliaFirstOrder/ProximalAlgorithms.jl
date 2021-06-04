using LinearAlgebra
using Test

using ProximalOperators
using ProximalAlgorithms

@testset "DRS/DRLS equivalence ($T)" for T in [Float32, Float64]
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]

    m, n = size(A)

    R = real(T)

    lam = R(0.1) * norm(A' * b, Inf)

    f = LeastSquares(A, b)
    g = NormL1(lam)

    x0 = zeros(R, n)

    drs_iter = ProximalAlgorithms.DouglasRachfordIteration(f=f, g=g, x0=x0, gamma=R(10) / opnorm(A)^2)
    drls_iter = ProximalAlgorithms.DRLSIteration(f=f, g=g, x0=x0, gamma=R(10) / opnorm(A)^2, lambda=R(1), c=-R(Inf), max_backtracks=1, H=ProximalAlgorithms.Noaccel())

    for (state_drs, state_drls) in Iterators.take(zip(drs_iter, drls_iter), 10)
        @test isapprox(state_drs.x, state_drls.xbar)
    end
end
