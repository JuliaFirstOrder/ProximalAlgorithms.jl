x0 = randn(10)
it, x, sol = ProximalAlgorithms.Template(x0)
@test it == 10
@test norm(x - x0) == 0.0
println(sol)
