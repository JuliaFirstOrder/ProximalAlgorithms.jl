# Solving LP with AFBA
#
#   minimize    c'x             -> f = <c,.>
#   subject to  Ax = b          -> h = ind{b}, L = A
#               x >= 0          -> g = ind{>=0}
#
# Dual LP
#
#   maximize    b'y
#   subject to  A'y <= c
#
# Optimality conditions
#
#   x >= 0              [primal feasibility 1]
#   Ax = b              [primal feasibility 2]
#   A'y <= c            [dual feasibility]
#   X'(c - A'y) = 0     [complementarity slackness]
#

Random.seed!(0)

n = 100 # primal dimension
m = 80 # dual dimension (i.e. number of linear equalities)
k = 50 # number of active dual constraints (must be 0 <= k <= n)
x_star = vcat(rand(k), zeros(n-k)) # primal optimal point
s_star = vcat(zeros(k), rand(n-k)) # dual optimal slack variable
y_star = randn(m) # dual optimal point

A = randn(m, n)
b = A*x_star
c = A'*y_star + s_star

using ProximalOperators

f = Linear(c)
g = IndNonnegative()
h = IndPoint(b)

using ProximalAlgorithms

x0 = zeros(n)
y0 = zeros(m)

it_afba, x_afba, y_afba, solver_afba = ProximalAlgorithms.AFBA(x0, y0; f=f, g=g, h=h, L=A, tol=1e-8, maxit=10000)

# Check and print solution quality measures
# (for some reason the returned dual iterate is the negative of the dual LP variable y)

TOL_ASSERT = 1e-6

meas_nonneg = -minimum(min.(0.0, x_afba))
println("Nonnegativity          : ", meas_nonneg)
@test meas_nonneg <= TOL_ASSERT

meas_feas_primal = norm(A*x_afba - b)
println("Primal feasibility     : ", meas_feas_primal)
@test meas_feas_primal <= TOL_ASSERT

meas_feas_dual = maximum(max.(0.0, -A'*y_afba - c))
println("Dual feasibility       : ", meas_feas_dual)
@test meas_feas_dual <= TOL_ASSERT

meas_compl = abs(dot(c + A'*y_afba, x_afba))
println("Complementarity        : ", meas_compl)
@test meas_compl <= TOL_ASSERT

println("Primal objective       : ", dot(c, x_afba))
println("Dual objective         : ", dot(b, -y_afba))
