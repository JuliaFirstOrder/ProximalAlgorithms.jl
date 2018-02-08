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

println("\n FPG ProximalAlgorithms.jl \n")
x0 = zeros(n)
@time itFBS, xFBS, sol = ProximalAlgorithms.FBS(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, fast = true, tol = 1e-8)

println("\n ZeroFPR ProximalAlgorithms.jl \n")
x0 = zeros(n)
@time itZ, xZ, sol = ProximalAlgorithms.ZeroFPR(x0; fq=f, Aq=A, g=g, gamma=1.0/norm(A)^2, tol = 1e-8)

@test norm(xFBS-xZ) < 1e-8
@test itZ < itFBS

