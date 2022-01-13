# # Getting started
#
# ## Installation
# 
# To install the package, hit `]` in the Julia REPL to enter the package manager mode, then run
# 
# ```julia
# pkg> add ProximalAlgorithms
# ```
# 
# to get the latest stable version, or
# 
# ```julia
# pkg> add ProximalAlgorithms#master
# ```
# 
# the latest development version (`master` branch).
#
# ## Overview
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
# To evaluate these first-order primitives, ProximalAlgorithms does the following:
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
# ## [Example: box constrained quadratic](@id box_qp)
# 
# As a four-line example, consider the minimization of a 2D quadratic function subject to box constraints:

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms

quadratic_cost(x) = dot([3.4 1.2; 1.2 89.1] * x, x) / 2 + dot([2.3, 99.9], x)
box_indicator = ProximalOperators.IndBox(-1, 1)

ffb = ProximalAlgorithms.FastForwardBackward(maxit=1000, tol=1e-5, verbose=true)
solution, iterations = ffb(zeros(2), f=quadratic_cost, g=box_indicator)

# Here, we defined the cost function `quadratic_cost`, and the constraint indicator `box_indicator`.
# Then we set up the optimization algorithm as the fast forward-backward splitting method (fast proximal gradient),
# with options for the maximum number of iterations, termination tolerance, verbosity.
# Finally, running the algorithm requires an initial point and the objective terms.
# 
# We can verify the correctness of the solution by checking that the negative gradient is orthogonal to the constraints, pointing outwards:

-ProximalAlgorithms.gradient(quadratic_cost, solution)[1]

# Or by plotting the solution against the cost function and constraint:

using Gadfly

contour = layer(
    z=(x,y) -> quadratic_cost([x, y]), xmin=[-2], xmax=[2], ymin=[-2], ymax=[2],
    Geom.contour(levels=[quadratic_cost(solution) + k * 12 for k in 0:100]),
)
point = layer(x=[solution[1]], y=[solution[2]])
box = layer(x=[-1, -1, 1, 1, -1], y=[1, -1, -1, 1, 1], Geom.path)
plot(contour, box, point, Guide.xlabel(nothing), Guide.ylabel(nothing))

# ## [Iterator interface](@id iterator_interface)
# 
# Under the hood, all algorithms are implemented as standard Julia iterators. In the above example,
# the `FastForwardBackward` algorithm internally uses a [`FastForwardBackwardIteration`](@ref ProximalAlgorithms.FastForwardBackwardIteration)
# iterator object.
# Each iterator outputs the complete iteration state: interacting directly with this object allows for more
# fine-grained inspection and control over the algorithm, for example to report the algorithm's progress in a
# custom way (cost decrease, or whatever other metric), or to interrupt the optimization based on a
# custom stopping criterion.
# 
#md # !!! note
#md #
#md #     Since each algorithm iterator type has its own logic, it will also have its own dedicated state structure.
#md #     Interacting with the state then requires being familiar with its structure, and with the nature of its attributes.
#md # 
# 
# For example, the optimization can be customized as follows: here we print the value of the cost function
# every ten iterations, and stop on a custom condition on the fixed-point residual norm.

ffbiter = ProximalAlgorithms.FastForwardBackwardIteration(x0=zeros(2), f=quadratic_cost, g=box_indicator)

for (k, state) in enumerate(ffbiter)
    if mod(k, 10) == 0
        println("iteration #$k: $(state.f_x)")
    end
    if norm(state.res) / state.gamma <= 1e-4
        break
    end
end

#md # !!! warning
#md #
#md #     To save on allocations, most (if not all) algorithms re-use state objects when iterating,
#md #     by updating the state *in place* instead of creating a new one. For this reason, one **should not**
#md #     - mutate the state object in any way, as this may corrupt the algorithm's logic,
#md #     - `collect` the sequence of states, since this will result in an array of identical objects.
#md #
# ## [Example: sparse linear regression](@id sparse_linreg)
# 
# Let's look at a least squares regression problem with L1 regularization:
# we will use the "diabetes dataset" (see [here](https://www4.stat.ncsu.edu/~boos/var.select/)), so let's start by loading the data.

using HTTP

splitlines(s) = split(s, "\n")
splitfields(s) = split(s, "\t")
parsefloat64(s) = parse(Float64, s)

function load_diabetes_dataset()
    res = HTTP.request("GET", "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt") 
    lines = res.body |> String |> strip |> splitlines
    return hcat((line |> splitfields .|> parsefloat64 for line in lines[2:end])...)'
end

data = load_diabetes_dataset()

training_input = data[1:end-100, 1:end-1]
training_label = data[1:end-100, end]

test_input = data[end-99:end, 1:end-1]
test_label = data[end-99:end, end]

n_training, n_features = size(training_input)

# Now we can set up the optimization problem we want to solve: we will minimize the mean squared error
# for a linear model, appropriately scaled so that the features in the training data are normally distributed
# ("standardization", this is known to help the optimization process).
# 
# After some simple manipulation, this standardized linear model can be implemented as follows:

using LinearAlgebra
using Statistics

input_loc = mean(training_input, dims=1) |> vec
input_scale = std(training_input, dims=1) |> vec

linear_model(wb, input) = input * wb[1:end-1] .+ wb[end]

function standardized_linear_model(wb, input)
    w_scaled = wb[1:end-1] ./ input_scale
    wb_scaled = vcat(w_scaled, wb[end] - dot(w_scaled, input_loc))
    return linear_model(wb_scaled, input)
end

# The loss term in the cost is then the following. Note that this is a regular Julia function:
# since the algorithm we will apply requires its gradient, automatic differentiation will
# do the work for us.

mean_squared_error(label, output) = mean((output .- label) .^ 2) / 2

training_loss(wb) = mean_squared_error(training_label, standardized_linear_model(wb, training_input))

# As regularization we will use the L1 norm, implemented in [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl):

using ProximalOperators

reg = ProximalOperators.NormL1(1)

# We want to minimize the sum of `training_loss` and `reg`, and for this task we can use `FastForwardBackward`,
# which implements the fast proximal gradient method (also known as fast forward-backward splitting, or FISTA).
# Therefore we construct the algorithm, then apply it to our problem by providing a starting point,
# and the objective terms `f=training_loss` (smooth) and `g=reg` (non smooth).

using ProximalAlgorithms

ffb = ProximalAlgorithms.FastForwardBackward()
solution, iterations = ffb(zeros(n_features + 1), f=training_loss, g=reg)

# We can now check how well the trained model performs on the test portion of our data.

test_output = standardized_linear_model(solution, test_input)
mean_squared_error(test_label, test_output)
