using Test
using LinearAlgebra
using ProximalOperators: NormL1
using ProximalAlgorithms
using Zygote
using ReverseDiff
using ForwardDiff
using AbstractDifferentiation: value_and_pullback_function
using AbstractDifferentiation: ZygoteBackend, ReverseDiffBackend, ForwardDiffBackend

@testset "Autodiff backend ($B on $T)" for (T, B) in Iterators.product(
    [Float32, Float64, ComplexF32, ComplexF64],
    [ZygoteBackend, ReverseDiffBackend, ForwardDiffBackend],
)
    if T <: Complex && B in [ReverseDiffBackend, ForwardDiffBackend]
        continue
    end

    R = real(T)
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]
    f = ProximalAlgorithms.AutoDifferentiable(x -> R(1/2) * norm(A * x - b, 2)^2, B())
    Lf = opnorm(A)^2
    m, n = size(A)

    x0 = zeros(T, n)

    f_x0, pb = value_and_pullback_function(B(), f, x0)
    grad_f_x0 = @inferred pb(one(R))[1]

    lam = R(0.1) * norm(A' * b, Inf)
    @test typeof(lam) == R
    g = NormL1(lam)
    x_star = T[-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]
    TOL = R(1e-4)
    solver = ProximalAlgorithms.FastForwardBackward(tol = TOL)
    x, it = solver(x0 = x0, f = f, g = g, Lf = Lf)
    @test eltype(x) == T
    @test norm(x - x_star, Inf) <= TOL
    @test it < 100
end
