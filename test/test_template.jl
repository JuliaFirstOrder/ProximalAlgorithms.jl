x0 = randn(10)

x1 = copy(x0)
sol1, it1 = ProximalAlgorithms.Template!(x1)
@test it1 == 10
@test norm(x0-x1) == 0.0

x2 = copy(x0)
sol2 = ProximalAlgorithms.TemplateSolver(maxit=20)
sol2, it2 = sol2(x2)
@test it2 == 20
@test norm(x0-x2) == 0.0
