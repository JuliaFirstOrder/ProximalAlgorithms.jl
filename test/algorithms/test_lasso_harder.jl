Random.seed!(123)

n = 2000
h = readdlm("h.txt")
A = Conv(Float64,(n,),h[:])
m = size(A,1)
#A = hcat([[zeros(i);h;zeros(n-1-i)] for i = 0:n-1]...) # Equivalent Full Matrix

x_star = Vector(sprandn(n,0.5))
b = A*x_star+20*randn(m)

f = Translate(SqrNormL2(), -b)
lam = 1e-1*norm(A'*b, Inf)
g = NormL1(lam)

## fast FBS/Adaptive
x0 = zeros(n)
@time it, xFBS, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, fast = true, tol = 1e-8)
@test it < 1000

# ZeroFPR/Adaptive
x0 = zeros(n)
@time it, xZ, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, tol = 1e-8)

@test norm(xFBS-xZ) < 1e-3
@test it < 70

# PANOC/Adaptive
x0 = zeros(n)
@time it, xP, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, tol = 1e-8)

@test norm(xP-xZ) < 1e-3
@test it < 140

#####################
# Complex Variables #
#####################

Random.seed!(123)
m, n = 10,100

A = randn(m,n)+im*randn(m,n)
U,S,V = svd(A)
#create ill conditioned matrix 
S[floor(Int,0.5*min(m,n)):end] .= 0.
A = U*diagm(0 => S)*V'

x_star = sprandn(n,0.5).+im.*sprandn(n,0.5)
b = A*x_star+20*(randn(m).+im*randn(m))

f = Translate(SqrNormL2(), -b)
f2 = LeastSquares(A, b)
lam = 1e-2*norm(A'*b, Inf)
g = NormL1(lam)

x0 = zeros(n)+im*zeros(n)
it, x_star, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, verbose = 0, tol = 1e-9)

# fast FBS/Adaptive
x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, fast = true, tol = 1e-6, maxit = 30000)

@test norm(x-x_star) < 1e-4
@test it < 25000

# ZeroFPR/Adaptive
x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, tol = 1e-6)

@test norm(x-x_star) < 1e-4
@test it < 700

# PANOC/Adaptive
x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, tol = 1e-6)

@test norm(x-x_star) < 1e-4
@test it < 1500
