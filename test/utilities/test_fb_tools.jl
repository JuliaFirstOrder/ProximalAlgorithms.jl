using Test
using LinearAlgebra
using ProximalCore: Zero
using ProximalAlgorithms: lower_bound_smoothness_constant, backtrack_stepsize!

@testset "Lipschitz constant estimation" for R in [Float32, Float64]

sv = R[0.01, 1.0, 1.0, 1.0, 100.0]
n = length(sv)
U, _ = qr(randn(n, n))
Q = U * Diagonal(sv) * U'
q = randn(n)

f(x) = 0.5 * dot(x, Q * x) + dot(q, x)
Lf = maximum(sv)
g = Zero()

for _ in 1:100
    x = randn(n)
    Lest = @inferred lower_bound_smoothness_constant(f, I, x)
    @test Lest <= Lf
end

x = randn(n)
Lest = @inferred lower_bound_smoothness_constant(f, I, x)
alpha = R(0.5)
gamma_init = 10 / Lest
gamma = gamma_init

for _ in 1:100
    x = randn(n)
    new_gamma, = @inferred backtrack_stepsize!(gamma, f, I, g, x, alpha=alpha)
    @test new_gamma <= gamma
    gamma = new_gamma
end

@test gamma < gamma_init

end
