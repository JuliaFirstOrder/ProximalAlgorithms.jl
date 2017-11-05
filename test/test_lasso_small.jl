using ProximalOperators

A = [  1.0  -2.0   3.0  -4.0  5.0;
       2.0  -1.0   0.0  -1.0  3.0;
      -1.0   0.0   4.0  -3.0  2.0;
      -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]

m, n = size(A)

f = Translate(SqrNormL2(), -b)
lam = 0.1*vecnorm(A'*b, Inf)
g = NormL1(lam)

x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

# Nonfast/Nonadaptive

x0 = zeros(n)
it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2)
x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2)
@test vecnorm(x - x_star, Inf) <= 1e-4
@test it == 140

# Nonfast/Adaptive

x0 = zeros(n)
it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true)
x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true)
@test vecnorm(x - x_star, Inf) <= 1e-4
@test it == 247

# Fast/Nonadaptive

x0 = zeros(n)
it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, fast=true)
x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, fast=true)
@test vecnorm(x - x_star, Inf) <= 1e-4
@test it == 83

# Fast/Adaptive

x0 = zeros(n)
it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true, fast=true)
x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true, fast=true)
@test vecnorm(x - x_star, Inf) <= 1e-4
@test it == 126
