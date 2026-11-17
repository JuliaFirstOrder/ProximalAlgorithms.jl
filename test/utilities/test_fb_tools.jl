using Test
using LinearAlgebra
using ProximalCore: Zero, gradient
using ProximalAlgorithms
using ProximalOperators: Quadratic
import ProximalAlgorithms: value_and_gradient

function value_and_gradient(f::Quadratic, x)
    y, fx = gradient(f, x)
    return fx, y
end

@testset "Lipschitz constant estimation" for R in [Float32, Float64]

    sv = R[0.01, 1.0, 1.0, 1.0, 100.0]
    n = length(sv)
    U, _ = qr(randn(R, n, n))
    Q = U * Diagonal(sv) * U'
    q = randn(R, n)

    f = Quadratic(Q, q)
    Lf = maximum(sv)
    g = Zero()

    for _ = 1:100
        x = randn(R, n)
        Lest = @inferred ProximalAlgorithms.lower_bound_smoothness_constant(f, I, x)
        @test typeof(Lest) == R
        @test Lest <= Lf
    end

    x = randn(n)
    Lest = @inferred ProximalAlgorithms.lower_bound_smoothness_constant(f, I, x)
    alpha = R(0.5)
    gamma_init = 10 / Lest
    gamma = gamma_init

    for _ = 1:100
        x = randn(n)
        new_gamma, = @inferred ProximalAlgorithms.backtrack_stepsize!(
            gamma,
            f,
            I,
            g,
            x,
            alpha = alpha,
        )
        @test new_gamma <= gamma
        gamma = new_gamma
    end

    @test gamma < gamma_init

end
