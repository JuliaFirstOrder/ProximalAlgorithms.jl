using ProximalOperators, ProximalAlgorithms

struct Rosenbrock{T}
  a::T
end
(f::Rosenbrock)(x) = (f.a-x[1])^2 + 100*(x[2]-x[1]^2)^2
function ProximalAlgorithms.value_and_gradient(f::Rosenbrock, x)
  return f(x), [2*(x[1]-f.a) + 400*(x[1]^2-x[2])*x[1], 200*(x[2]-x[1]^2)]
end

f = Rosenbrock(1.0)
g = NormL1(2.0)

solver = ProximalAlgorithms.FastForwardBackward(tol = 1e-5, verbose = true)
x, iters = solver(x0 = ones(2), f = f, g = g)

using Zygote
using DifferentiationInterface: AutoZygote

f_auto = ProximalAlgorithms.AutoDifferentiable(x -> f(x), AutoZygote())
x, iters = solver(x0 = ones(2), f = f_auto, g = g)