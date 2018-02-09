using AbstractOperators
using ProximalOperators

A1 = [ 1.0  -2.0   3.0  -4.0  5.0;
       2.0  -1.0   0.0  -1.0  3.0]
A2 = [-1.0   0.0   4.0  -3.0  2.0;
      -1.0  -1.0  -1.0   1.0  3.0]
A = [A1; A2]
opA = VCAT(MatrixOp(A1), MatrixOp(A2))

b1 = [1.0, 2.0]
b2 = [3.0, 4.0]
b = [b1; b2]

f = SeparableSum(Translate(SqrNormL2(), -b1), Translate(SqrNormL2(), -b2))
lam = 0.1*vecnorm(A'*b, Inf)
g = NormL1(lam)

x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

# Nonfast/Nonadaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, gamma=1.0/norm(A)^2)
@test vecnorm(x .- x_star, Inf) <= 1e-4
#@test it == 140
println(sol)

# Nonfast/Adaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, adaptive=true)
@test vecnorm(x .- x_star, Inf) <= 1e-4
#@test it == 247
println(sol)

# Fast/Nonadaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, gamma=1.0/norm(A)^2, fast=true)
@test vecnorm(x .- x_star, Inf) <= 1e-4
#@test it == 94
println(sol)

# Fast/Adaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, adaptive=true, fast=true)
@test vecnorm(x .- x_star, Inf) <= 1e-4
#@test it == 156
println(sol)

# ZeroFPR/Nonadaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=opA, g=g, gamma=1.0/norm(A)^2)
@test vecnorm(x - x_star, Inf) <= 1e-4
#@test it == 8
println(sol)

# ZeroFPR/Adaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=opA, g=g, adaptive=true)
@test vecnorm(x - x_star, Inf) <= 1e-4
#@test it == 10
println(sol)

# PANOC/Nonadaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=opA, g=g, gamma=1.0/norm(A)^2)
@test vecnorm(x - x_star, Inf) <= 1e-4
#@test it == 8
println(sol)

# PANOC/Adaptive

x0 = ProximalAlgorithms.blockzeros(x_star)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=opA, g=g, adaptive=true)
@test vecnorm(x - x_star, Inf) <= 1e-4
#@test it == 10
println(sol)
