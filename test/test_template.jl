x0 = randn(10)
it, x, sol = ProximalAlgorithms.Template(x0)
@test it == 10
@test norm(x - x0) == 0.0

x0 = randn(10)
call = ProximalAlgorithms.TemplateSolver(maxit=20)
it, x, sol = call(x0)
@test it == 20
@test norm(x - x0) == 0.0
