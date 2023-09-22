# # [Sparse linear regression](@id sparse_linreg)
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

using ProximalAlgorithms

training_loss = ProximalAlgorithms.ZygoteFunction(
    wb -> mean_squared_error(training_label, standardized_linear_model(wb, training_input))
)

# As regularization we will use the L1 norm, implemented in [ProximalOperators](https://github.com/JuliaFirstOrder/ProximalOperators.jl):

using ProximalOperators

reg = ProximalOperators.NormL1(1)

# We want to minimize the sum of `training_loss` and `reg`, and for this task we can use `FastForwardBackward`,
# which implements the fast proximal gradient method (also known as fast forward-backward splitting, or FISTA).
# Therefore we construct the algorithm, then apply it to our problem by providing a starting point,
# and the objective terms `f=training_loss` (smooth) and `g=reg` (non smooth).

ffb = ProximalAlgorithms.FastForwardBackward()
solution, iterations = ffb(x0=zeros(n_features + 1), f=training_loss, g=reg)

# We can now check how well the trained model performs on the test portion of our data.

test_output = standardized_linear_model(solution, test_input)
mean_squared_error(test_label, test_output)
