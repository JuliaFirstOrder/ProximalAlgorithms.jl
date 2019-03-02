using AbstractOperators, RecursiveArrayTools
using ProximalOperators

A1 = [ 1.0  -2.0   3.0;
       2.0  -1.0   0.0;
      -1.0   0.0   4.0;
      -1.0  -1.0  -1.0]
A2 = [-4.0  5.0;
	  -1.0  3.0;
	  -3.0  2.0;
	   1.0  3.0]
A = [A1 A2]

opA1 = MatrixOp(A1)
opA2 = MatrixOp(A2)
opA = HCAT(opA1, opA2)

b = [1.0, 2.0, 3.0, 4.0]

m, n = size(A)

f = Translate(SqrNormL2(), -b)
lam = 0.1*norm(A'*b, Inf)
g = SeparableSum(NormL1(lam), NormL1(lam))

x_star = ArrayPartition([-3.877278911564627e-01, 0, 0], [2.174149659863943e-02, 6.168435374149660e-01])

# Nonfast/Nonadaptive

x0 = zero(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, gamma=1.0/opnorm(A)^2)
@test maximum(abs,x .- x_star) <= 1e-4
@test it < 150
println(sol)

# Nonfast/Adaptive

x0 = zero(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, adaptive=true)
@test maximum(abs,x .- x_star) <= 1e-4
@test it < 300
println(sol)

# Fast/Nonadaptive

x0 = zero(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, gamma=1.0/opnorm(A)^2, fast=true)
@test maximum(abs, x .- x_star) <= 1e-4
@test it < 100
println(sol)

# Fast/Adaptive

x0 = zero(x_star)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=opA, g=g, adaptive=true, fast=true)
@test maximum(abs,x .- x_star) <= 1e-4
@test it < 200
println(sol)

# ZeroFPR/Adaptive

x0 = zero(x_star)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=opA, g=g, adaptive=true)
@test maximum(abs,x .- x_star) <= 1e-4
@test it < 15
println(sol)

# PANOC/Adaptive

x0 = zero(x_star)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=opA, g=g, adaptive=true)
@test maximum(abs,x .- x_star) <= 1e-4
@test it < 20
println(sol)
