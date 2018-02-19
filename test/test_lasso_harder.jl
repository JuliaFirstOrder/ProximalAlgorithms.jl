using ProximalOperators

srand(123)
m, n = 10,100

A = randn(m,n)
U,S,V = svd(A)
#create ill conditioned matrix 
S[floor(Int,0.5*min(m,n)):end] .= 0.
A = U*diagm(S)*V'

x_star = sprandn(n,0.5)
b = A*x_star+20*randn(m)

f = Translate(SqrNormL2(), -b)
f2 = LeastSquares(A, b)
lam = 1e-2*vecnorm(A'*b, Inf)
g = NormL1(lam)

# fast FBS/Nonadaptive
x0 = zeros(n)
@time it, xFBS, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, fast = true, tol = 1e-8)
@test it < 1800

# ZeroFPR/Nonadaptive
x0 = zeros(n)
@time it, xZ, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, tol = 1e-8)

@test norm(xFBS-xZ) < 1e-8
@test it < 150

# PANOC/Nonadaptive
x0 = zeros(n)
@time it, xP, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, tol = 1e-8)

@test norm(xP-xZ) < 1e-8
@test it < 300

#####################
# Complex Variables #
#####################

srand(123)
m, n = 10,100

A = randn(m,n)+im*randn(m,n)
U,S,V = svd(A)
#create ill conditioned matrix 
S[floor(Int,0.5*min(m,n)):end] .= 0.
A = U*diagm(S)*V'

x_star = sprandn(n,0.5).+im.*sprandn(n,0.5)
b = A*x_star+20*(randn(m).+im*randn(m))

f = Translate(SqrNormL2(), -b)
f2 = LeastSquares(A, b)
lam = 1e-2*vecnorm(A'*b, Inf)
g = NormL1(lam)

x0 = zeros(n)+im*zeros(n)
it, x_star, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, verbose = 0, tol = 1e-9)

# fast FBS/Adaptive
x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, fast = true, tol = 1e-6, maxit = 30000)

@test norm(x-x_star) < 1e-4
@test it < 23000

# ZeroFPR/Adaptive
x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, tol = 1e-6)

@test norm(x-x_star) < 1e-4
@test it < 400

# PANOC/Adaptive
x0 = zeros(n)+im*zeros(n)
@time it, x, sol = ProximalAlgorithms.PANOC(x0; fq=f, Aq=A, g=g, tol = 1e-6)

@test norm(x-x_star) < 1e-4
@test it < 1200
