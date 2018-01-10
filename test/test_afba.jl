using ProximalOperators
using ProximalAlgorithms

# testing combinations with two terms

A = [  1.0  -2.0   3.0  -4.0  5.0;
       2.0  -1.0   0.0  -1.0  3.0;
      -1.0   0.0   4.0  -3.0  2.0;
      -1.0  -1.0  -1.0   1.0  3.0]
b = [1.0, 2.0, 3.0, 4.0]

m, n = size(A)

f = Translate(SqrNormL2(), -b)
f2 = LeastSquares(A, b)
lam = 0.1*vecnorm(A'*b, Inf)
g = NormL1(lam)

x_star = [-3.877278911564627e-01, 0, 0, 2.174149659863943e-02, 6.168435374149660e-01]





stuff = [
  Dict( "theta"      => 2,
        "mu"         => 0,
      ),
  Dict( "theta"      => 1,
        "mu"         => 1,
      ),
  Dict( "theta"      => 0,
        "mu"         => 1,
      ),
  Dict( "theta"      => 0,
        "mu"         => 0,
      ),
  Dict( "theta"      => 1,
        "mu"         => 0,
      )
  ]



srand(2)


for i = 1:length(stuff)

theta = stuff[i]["theta"]
mu    = stuff[i]["mu"]

x0 = randn(n)
y0 = randn(m) 
#h\equiv 0 (FBS)

@time it, x, sol = ProximalAlgorithms.AFBA(x0,y0; g=g,f=f2, betaQ =norm(A'*A), theta=theta, mu=mu)   
println("      nnz(x)    = $(norm(sol.x, 2))")
@test vecnorm(x - x_star, Inf) <= 1e-4
println(sol)

# f=\equiv 0 (Chambolle-Pock)
@time it, x, sol = ProximalAlgorithms.AFBA(x0,y0; g=g, h=f, L=A, theta=theta, mu=mu)  
@test vecnorm(x - x_star, Inf) <= 1e-4
println(sol)

# g\equiv 0
x0 = randn(n)
y0 = randn(n) # since L= Identity 
@time it, x, sol = ProximalAlgorithms.AFBA(x0,y0; h=g,f=f2, betaQ =norm(A'*A),  theta=theta, mu=mu)  
@test vecnorm(x - x_star, Inf) <= 1e-4
println(sol)

end 