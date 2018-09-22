A = [  1.0  -2.0   3.0  -4.0  5.0;
       2.0  -1.0   0.0  -1.0  3.0;
      -1.0   0.0   4.0  -3.0  2.0;
      -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]

m, n = size(A)

f = Translate(SqrNormL2(), -b)
f2 = LeastSquares(A, b)
lam = 0.1*norm(A'*b, Inf)
g = NormL1(lam)

x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]

## Nonfast/Nonadaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 150 
println(sol)

# testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

# Nonfast/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 300
println(sol)

# Fast/Nonadaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2, fast=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 100
println(sol)

# testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

# Fast/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true, fast=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 200
println(sol)

# ZeroFPR/Nonadaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 15
println(sol)

#testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

# ZeroFPR/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 15

# PANOC/Nonadaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 20
println(sol)

# testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

## PANOC/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 20
println(sol)

# Douglas-Rachford

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.DRS(x0; f=f2, g=g, gamma=10.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 30
println(sol)

#testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

#####################
# Complex Variables #
#####################

Random.seed!(123)
m, n = 10,5

A = randn(m,n)+im*randn(m,n)
b = randn(m)+im*randn(m)

f = Translate(SqrNormL2(), -b)
f2 = LeastSquares(A, b)
lam = 0.01*norm(A'*b, Inf)
g = NormL1(lam)

x0 = zeros(n)+im*zeros(n)
it, x_star, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2, verbose = 0)

## Nonfast/Nonadaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 200
println(sol)

# testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

# Nonfast/Adaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 250
println(sol)

# Fast/Nonadaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2, fast=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 120
println(sol)

# testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

# Fast/Adaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, adaptive=true, fast=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 120
println(sol)

# ZeroFPR/Nonadaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 15
println(sol)

#testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

# ZeroFPR/Adaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 15

# PANOC/Nonadaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, gamma=1.0/opnorm(A)^2)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 20
println(sol)

# testing solver already at solution
@time it, x = ProximalAlgorithms.run!(sol)
@test it == 1

## PANOC/Adaptive

x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 20
println(sol)

#############################################
# Real Variables  mapped to Complex Numbers #
#############################################

Random.seed!(123)
n = 2^6
A = AbstractOperators.DFT(n)[1:div(n,2)]      # overcomplete dictionary 

x = sprandn(n,0.5)
b = fft(x)[1:div(n,2)]

#f = Translate(LogisticLoss(ones(n)), -b)
f = Translate(SqrNormL2(), -b)
lam = 0.01*norm(A'*b,Inf)
g = NormL1(lam)

x0 = zeros(n)
it, x_star, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, verbose = 0, tol = 1e-8)

# Nonfast/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 50
println(sol)

# Fast/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true, fast=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 50
println(sol)

# ZeroFPR/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 15
println(sol)

# PANOC/Adaptive

x0 = zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, tol=1e-6, adaptive=true)
@test norm(x - x_star, Inf) <= 1e-4
@test it < 20
println(sol)
