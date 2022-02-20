# # Getting started
#
# The methods implemented in ProximalAlgorithms are commonly referred to as (you've guessed it) *proximal algorithms*,
# in that they rely on the *proximal operator* (or *mapping*) to deal with non-differentiable terms in the objective.
# Loosely speaking, the algorithms in this package can be used to solve problems of the form
# 
# ```math
# \operatorname*{minimize}_x\ \sum_{i=1}^N f_i(x)
# ```
# 
# where ``N`` depends on the specific algorithm, together with specific assumptions on the terms ``f_i`` (like smoothness, convexity, strong convexity).
# The problem above is solved by iteratively accessing specific *first order information* on the terms ``f_i``,
# like their gradient ``\nabla f_i`` or their proximal mapping ``\operatorname{prox}_{f_i}``:
# ```math
# \mathrm{prox}_{\gamma f_i}(x) = \arg\min_z \left\{ f_i(z) + \tfrac{1}{2\gamma}\|z-x\|^2 \right\}
# ```
# The literature on proximal operators and algorithms is vast: for an overview, one can refer to [Parikh2014](@cite), [Beck2017](@cite).
# 
# To evaluate these first-order primitives, in ProximalAlgorithms:
# * ``\nabla f_i`` falls back to using automatic differentiation (as provided by [Zygote](https://github.com/FluxML/Zygote.jl)).
# * ``\operatorname{prox}_{f_i}`` relies on the intereface of [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl).
# Both of the above can be implemented for custom function types, as [documented here](@ref custom_terms).
# 
#md # !!! note
#md # 
#md #     Each of the [implemented algorithms](@ref problems_algorithms) assumes a different structure of the objective to be optimized
#md #     (e.g., a specific number ``N`` of terms), with specific assumptions (e.g., smoothness, convexity).
#md #     Furthermore, multiple algorithms can often be applied to the same problem (possibly through appropriate reformulation) and are expected
#md #     to perform differently; sometimes, even the *same algorithm* can be applied in multiple ways to the same problem, by grouping (*splitting*)
#md #     the terms in different ways.
#md # 
#md #     Because of these reasons, ProximalAlgorithms **does not offer a modeling language** to automagically minimize any objective:
#md #     rather, the user is expected to formulate their problem by providing the right objective terms to the algorithm of choice.
#md #     Please refer to the [this section of the manual](@ref problems_algorithms) for information on what terms can be provided and under which assumptions.
# 
# ## [Interface to algorithms](@id algorithm_interface)
# 
# At a high level, using algorithms from ProximalAlgorithms amounts to the following.
# 1. Instantiate the algorithm, with options like the termination tolerance, verbosity level, or other algorithm-specific parameters.
# 2. Call the algorithm on the problem description: this amounts to the initial point, the objective terms, and possibly additional required information (e.g. Lipschitz constants).
# 
# See [here](@ref problems_algorithms) for the list of available algorithm constructors, for different types of problems.
# In general however, algorithms are instances of the [`IterativeAlgorithm`](@ref) type.
# 
# ## [Example: box constrained quadratic](@id box_qp)
# 
# As a simple example, consider the minimization of a 2D quadratic function subject to box constraints,
# which we will solve using the fast proximal gradient method (also known as fast forward-backward splitting):

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms

quadratic_cost(x) = dot([3.4 1.2; 1.2 89.1] * x, x) / 2 + dot([-2.3, 99.9], x)
box_indicator = ProximalOperators.IndBox(0, 1)

ffb = ProximalAlgorithms.FastForwardBackward(maxit=1000, tol=1e-5, verbose=true)

# Here, we defined the cost function `quadratic_cost`, and the constraint indicator `box_indicator`.
# Then we set up the optimization algorithm of choice, [`FastForwardBackward`](@ref),
# with options for the maximum number of iterations, termination tolerance, verbosity.
# Finally, we run the algorithm by providing an initial point and the objective terms defining the problem:

solution, iterations = ffb(x0=ones(2), f=quadratic_cost, g=box_indicator)

# We can verify the correctness of the solution by checking that the negative gradient is orthogonal to the constraints, pointing outwards:

-ProximalAlgorithms.gradient(quadratic_cost, solution)[1]

# Or by plotting the solution against the cost function and constraint:

using Plots

contour(-1:0.1:2, -1:0.1:2, (x,y) -> quadratic_cost([x, y]), fill=true, framestyle=:none, background=nothing)
plot!(Shape([0, 1, 1, 0], [0, 0, 1, 1]), opacity=.5, label="feasible set")
scatter!([solution[1]], [solution[2]], color=:red, markershape=:star5, label="computed solution")

# ## [Iterator interface](@id iterator_interface)
# 
# Under the hood, algorithms are implemented in the form of standard Julia iterators:
# constructing such iterator objects directly, and looping over them, allows for more fine-grained control over the termination condition,
# or what information from the iterations get logged.
# 
# Each iterator is constructed with the full problem description (objective terms and, if needed, additional information like Lipschitz constats)
# and algorithm options (usually step sizes, and any other parameter or option of the algorithm),
# and produces the sequence of states of the algorithm, so that one can do (almost) anything with it.
# 
#md # !!! note 
#md #     Iterators only implement the algorithm iteration logic, and not additional details like stopping criteria.
#md #     As such, iterators usually yield an infinite sequence of states:
#md #     when looping over them, be careful to properly guard the loop with a stopping criterion.
# 
#md # !!! warning
#md #
#md #     To save on allocations, most (if not all) algorithms re-use state objects when iterating,
#md #     by updating the state *in place* instead of creating a new one. For this reason:
#md #     - one **should not** mutate the state object in any way, as this may corrupt the algorithm's logic;
#md #     - one **should not** `collect` the sequence of states, since this will result in an array of identical objects.
#md #
# 
# Iterator types are named after the algorithm they implement, so the relationship should be obvious:
# * the `ForwardBackward` algorithm uses the `ForwardBackwardIteration` iterator type;
# * the `FastForwardBackward` algorithm uses the `FastForwardBackwardIteration` iterator type;
# * the `DouglasRachford` algorithm uses the `DouglasRachfordIteration` iterator type;
# and so on.
# 
# Let's see what this means in terms of the previous example.
# 
# ## [Example: box constrained quadratic (cont)](@id box_qp_cont)
# 
# Let's solve the problem from the [previous example](@ref box_qp) by directly interacting with the underlying iterator: 
# the `FastForwardBackward` algorithm internally uses a [`FastForwardBackwardIteration`](@ref) object.

ffbiter = ProximalAlgorithms.FastForwardBackwardIteration(x0=ones(2), f=quadratic_cost, g=box_indicator)

# We can now perform anything we want throughout the iteration, by just looping over the iterator:
# for example, we can store the sequence of iterates from the algorithm, to later plot them,
# and stop whenever two successive iterates are closer than a given tolerance.

xs = []
for state in ffbiter
    push!(xs, copy(state.x))
    if length(xs) > 1 && norm(xs[end] - xs[end - 1]) / (1 + norm(xs[end])) <= 1e-5
        break
    end
end

contour(-1:0.1:2, -1:0.1:2, (x,y) -> quadratic_cost([x, y]), fill=true, framestyle=:none, background=nothing)
plot!(Shape([0, 1, 1, 0], [0, 0, 1, 1]), opacity=.5, label="feasible set")
plot!([x[1] for x in xs], [x[2] for x in xs], markershape=:circle, label="algorithm trajectory")
scatter!([solution[1]], [solution[2]], color=:red, markershape=:star5, label="computed solution")

#md # !!! note
#md #
#md #     Since each algorithm iterator type has its own logic, it will also have its own dedicated state structure.
#md #     Interacting with the state then requires being familiar with its structure, and with the nature of its attributes.
#md # 
