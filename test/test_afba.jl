using ProximalOperators
using ProximalAlgorithms

### this test includes two tests

## 1-testing combinations with two terms: Lasso

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


stuff = [
  Dict( "theta"      => 2,
        "mu"         => 0,
        "it"         => (70,135,100),
      ),
  Dict( "theta"      => 1,
        "mu"         => 1,
        "it"         => (80,130,15),
      ),
  Dict( "theta"      => 0,
        "mu"         => 1,
        "it"         => (90,345,120),
      ),
  Dict( "theta"      => 0,
        "mu"         => 0,
        "it"         => (80,170,180),
      ),
  Dict( "theta"      => 1,
        "mu"         => 0,
        "it"         => (90,130,230),
      )
  ]

Random.seed!(0)

for i = 1:length(stuff)

theta = stuff[i]["theta"]
mu    = stuff[i]["mu"]
itnum = stuff[i]["it"]

x0 = randn(n)

# h\equiv 0 (FBS)
y0 = randn(n)
@time it, x, y, sol = ProximalAlgorithms.AFBA(x0, y0; g=g, f=f2, betaQ =opnorm(A'*A), theta=theta, mu=mu)
println("      nnz(x)    = $(norm(sol.x, 2))")
@test norm(x - x_star, Inf) <= 1e-4
@test it <=itnum[1]
println(sol)

@time it, (x, y) = ProximalAlgorithms.run!(sol)

# f=\equiv 0 (Chambolle-Pock)
y0 = randn(m)
@time it, x, y, sol = ProximalAlgorithms.AFBA(x0, y0; g=g, h=f, L=A, theta=theta, mu=mu)
@test norm(x - x_star, Inf) <= 1e-4
@test it <=itnum[2]
println(sol)

# g\equiv 0
y0 = randn(n) # since L= Identity
@time it, x, y, sol = ProximalAlgorithms.AFBA(x0, y0; h=g, f=f2, betaQ=opnorm(A'*A), theta=theta, mu=mu)
@test norm(x - x_star, Inf) <= 1e-4
@test it <=itnum[3]
println(sol)

end

## 2- testing with three terms: 1/2\|Ax-b\|^2+ λ\|x\|_1 + + λ_2*\|x\|^2

lam2= 1;


x0 = randn(n)
y0 = randn(n)
itnum= ((130,130),(40,40),(90,90),(150,150),(170,170)); # the number of iterations

for i = 1:length(stuff)
theta = stuff[i]["theta"]
mu    = stuff[i]["mu"]
@time it, x, y, sol = ProximalAlgorithms.AFBA(x0,y0; g=g,f=f2, h = SqrNormL2(lam2), betaQ =opnorm(A'*A), theta=theta, mu=mu)
println("      nnz(x)    = $(norm(sol.x, 2))")

 # the optimality conditions
temp = lam*sign.(x) + A'*(A*x-b) + lam2*x
ind= findall(x -> abs.(x)<1e-8,x)
t1= length(findall((temp[ind] .<= lam .* temp[ind] .>=-lam)==false))
t2= length(findall((abs.(deleteat!(temp,ind)) .<1e-8)==false))
@test t1+t2 == 0
@test it <=itnum[i][1]
println(sol)


@time it, x, y, sol = ProximalAlgorithms.AFBA(x0,y0; h=g,f=f2, g = SqrNormL2(lam2), betaQ =opnorm(A'*A), theta=theta, mu=mu)
println("      nnz(x)    = $(norm(sol.x, 2))")
 # the optimality conditions
temp = lam*sign.(x) + A'*(A*x-b) + lam2*x
ind= findall(x -> abs.(x)<1e-8,x)
t1= length(findall((temp[ind] .<= lam .* temp[ind] .>=-lam)==false))
t2= length(findall((abs.(deleteat!(temp,ind)) .<1e-8)==false))
@test t1+t2 == 0
@test it <=itnum[i][2]
println(sol)

end

# TODO: add test including function l
