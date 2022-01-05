using LinearAlgebra
using Test

using ProximalOperators
using ProximalAlgorithms:
    DouglasRachfordIteration, DRLSIteration,
    ForwardBackwardIteration, PANOCIteration,
    NOLIPIteration,
    NoAcceleration

@testset "DR/DRLS equivalence ($T)" for T in [Float32, Float64]
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

    dr_iter = DouglasRachfordIteration(f=f, g=g, x0=x0, gamma=R(10) / opnorm(A)^2)
    drls_iter = DRLSIteration(f=f, g=g, x0=x0, gamma=R(10) / opnorm(A)^2, lambda=R(1), c=-R(Inf), max_backtracks=1, directions=NoAcceleration())

    for (dr_state, drls_state) in Iterators.take(zip(dr_iter, drls_iter), 10)
        @test isapprox(dr_state.x, drls_state.xbar)
    end
end

@testset "FB/PANOC equivalence ($T)" for T in [Float32, Float64]
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

    fb_iter = ForwardBackwardIteration(f=f, g=g, x0=x0, gamma=R(0.95) / opnorm(A)^2)
    panoc_iter = PANOCIteration(f=f, g=g, x0=x0, gamma=R(0.95) / opnorm(A)^2, max_backtracks=1, directions=NoAcceleration())

    for (fb_state, panoc_state) in Iterators.take(zip(fb_iter, panoc_iter), 10)
        @test isapprox(fb_state.z, panoc_state.z)
    end
end

@testset "PANOC/NOLIP equivalence ($T)" for T in [Float32, Float64]
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

    panoc_iter = PANOCIteration(f=f, g=g, x0=x0, gamma=R(0.95) / opnorm(A)^2)
    nolip_iter = NOLIPIteration(f=f, g=g, x0=x0, gamma=R(0.95) / opnorm(A)^2)

    for (panoc_state, nolip_state) in Iterators.take(zip(panoc_iter, nolip_iter), 10)
        @test isapprox(panoc_state.z, nolip_state.z)
    end
end
