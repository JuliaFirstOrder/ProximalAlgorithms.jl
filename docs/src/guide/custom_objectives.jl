# ```@meta
# CurrentModule = ProximalAlgorithms
# ```
# # [Custom objective terms](@id custom_terms)
# 
# ProximalAlgorithms relies on the first-order primitives defined in [ProximalCore](https://github.com/JuliaFirstOrder/ProximalCore.jl).
# While a rich library of function types, implementing such primitives, is provided by [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl),
# one may need to formulate problems using custom objective terms.
# When that is the case, one only needs to implement the right first-order primitive,
# ``\nabla f`` or ``\operatorname{prox}_{\gamma f}`` or both, for algorithms to be able
# to work with ``f``.
# 
# Defining the proximal mapping for a custom function type requires adding a method for [`ProximalCore.prox!`](@ref).
# 
# To compute gradients, algorithms use [`value_and_gradient`](@ref):
# this relies on [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl), for automatic differentiation
# with any of its supported backends, when functions are wrapped in [`AutoDifferentiable`](@ref),
# as the examples below show.
# 
# If however you would like to provide your own gradient implementation (e.g. for efficiency reasons),
# you can simply implement a method for [`value_and_gradient`](@ref) on your own function type.
# 
# ```@docs
# ProximalCore.prox
# ProximalCore.prox!
# ProximalAlgorithms.value_and_gradient
# ProximalAlgorithms.AutoDifferentiable
# ```
# 
# ## Example: constrained Rosenbrock
# 
# Let's try to minimize the celebrated Rosenbrock function, but constrained to the unit norm ball. The cost function is

using Zygote
using DifferentiationInterface: AutoZygote
using ProximalAlgorithms

rosenbrock2D = ProximalAlgorithms.AutoDifferentiable(
    x -> 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2,
    AutoZygote(),
)

# To enforce the constraint, we define the indicator of the unit ball, together with its proximal mapping:
# this is simply projection onto the unit norm ball, so it is sufficient to normalize any given point that lies
# outside of the set.

using LinearAlgebra
using ProximalCore

struct IndUnitBall end

(::IndUnitBall)(x) = norm(x) > 1 ? eltype(x)(Inf) : eltype(x)(0)

function ProximalCore.prox!(y, ::IndUnitBall, x, gamma)
    if norm(x) > 1
        y .= x ./ norm(x)
    else
        y .= x
    end
    return zero(eltype(x))
end

# We can now minimize the function, for which we will use [`PANOC`](@ref), which is a Newton-type method:

panoc = ProximalAlgorithms.PANOC()
solution, iterations = panoc(x0 = -ones(2), f = rosenbrock2D, g = IndUnitBall())

# Plotting the solution against the cost function contour and constraint, gives an idea of its correctness.

using Plots

contour(
    -2:0.1:2,
    -2:0.1:2,
    (x, y) -> rosenbrock2D([x, y]),
    fill = true,
    framestyle = :none,
    background = nothing,
)
plot!(Shape(cos.(0:0.01:2*pi), sin.(0:0.01:2*pi)), opacity = 0.5, label = "feasible set")
scatter!(
    [solution[1]],
    [solution[2]],
    color = :red,
    markershape = :star5,
    label = "computed solution",
)

# ## Example: counting operations
# 
# It is often interesting to measure how many operations (gradient- or prox-evaluation) an algorithm is taking.
# In fact, in algorithms involving backtracking or some other line-search logic, the iteration count may not be entirely
# representative of the amount of operations are being performed; or maybe some specific implementations require
# additional operations to be performed when checking stopping conditions. All of this makes it difficult to quantify the exact
# iteration complexity.
# 
# We can achieve this by wrapping functions in a dedicated `Counting` type:

mutable struct Counting{T}
    f::T
    eval_count::Int
    gradient_count::Int
    prox_count::Int
end

Counting(f::T) where {T} = Counting{T}(f, 0, 0, 0)

function (f::Counting)(x)
    f.eval_count += 1
    return f.f(x)
end

# Now we only need to intercept any call to [`value_and_gradient`](@ref) and [`prox!`](@ref) and increase counters there:

function ProximalAlgorithms.value_and_gradient(f::Counting, x)
    f.eval_count += 1
    f.gradient_count += 1
    return ProximalAlgorithms.value_and_gradient(f.f, x)
end

function ProximalCore.prox!(y, f::Counting, x, gamma)
    f.prox_count += 1
    return ProximalCore.prox!(y, f.f, x, gamma)
end

# We can run again the previous example, this time wrapping the objective terms within `Counting`:

f = Counting(rosenbrock2D)
g = Counting(IndUnitBall())

solution, iterations = panoc(x0 = -ones(2), f = f, g = g)

# and check how many operations where actually performed:

println("function evals: $(f.eval_count)")
println("gradient evals: $(f.gradient_count)")
println("    prox evals: $(g.prox_count)")
