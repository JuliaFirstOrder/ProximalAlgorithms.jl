# # Custom objective terms
# 
# ProximalAlgorithms relies on the first-order primitives implemented in [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl):
# while a rich library of function types is provided there, one may need to formulate
# problems using custom objective terms.
# When that is the case, one only needs to implement the right first-order primitive,
# ``\nabla f`` or ``\operatorname{prox}_{\gamma f}`` or both, for algorithms to be able
# to work with ``f``.
# 
# Defining the proximal mapping for a custom function type requires adding a method for [`prox!`](@ref ProximalAlgorithms.prox!).
# 
# For computing gradients, ProximalAlgorithms provides a fallback definition for [`gradient!`](@ref ProximalAlgorithms.gradient!), 
# relying on [Zygote](https://github.com/FluxML/Zygote.jl) to use automatic differentiation.
# Therefore, you can provide any (differentiable) Julia function wherever gradients need to be taken,
# and everything will work out of the box.
# 
# If however one would like to provide their own gradient implementation (e.g. for efficiency reasons),
# they can simply implement a method for [`gradient!`](@ref ProximalAlgorithms.gradient!).
# 
# ```@docs
# ProximalAlgorithms.prox!(y, f, x, gamma)
# ProximalAlgorithms.gradient!(g, f, x)
# ```
# 
# ## Example: constrained Rosenbrock
# 
# Let's try to minimize the celebrated Rosenbrock function, but constrained to the unit norm ball. The cost function is

rosenbrock2D(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

# To enforce the constraint, we define the indicator of the unit ball, together with its proximal mapping:
# this is simply projection onto the unit norm ball, so it is sufficient to normalize any given point that lies
# outside of the set.

using LinearAlgebra
using ProximalOperators

struct IndUnitBall <: ProximalOperators.ProximableFunction end

(::IndUnitBall)(x) = norm(x) > 1 ? eltype(x)(Inf) : eltype(x)(0)

function ProximalOperators.prox!(y, ::IndUnitBall, x, gamma)
    if norm(x) > 1
        y .= x ./ norm(x)
    else
        y .= x
    end
    return zero(eltype(x))
end

# We can now minimize the function, for which we will use `PANOC`, which is a Newton-type method:

using ProximalAlgorithms

panoc = ProximalAlgorithms.PANOC()
solution, iterations = panoc(-ones(2), f=rosenbrock2D, g=IndUnitBall())

# Plotting the solution against the cost function contour and constraint, gives an idea of its correctness.

using Gadfly

contour = layer(
    z=(x,y) -> rosenbrock2D([x, y]), xmin=[-2], xmax=[2], ymin=[-2], ymax=[2],
    Geom.contour(levels=vcat([1.0, 10.0], [100.0 + 200.0 * k for k in 0:30])),
)
point = layer(x=[solution[1]], y=[solution[2]], Geom.point)
circle = layer(x=cos.(0:0.01:2*pi), y=sin.(0:0.01:2*pi), Geom.path)
plot(contour, circle, point, Guide.xlabel(nothing), Guide.ylabel(nothing))