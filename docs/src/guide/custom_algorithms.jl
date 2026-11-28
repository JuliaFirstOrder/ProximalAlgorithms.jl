# ```@meta
# CurrentModule = ProximalAlgorithms
# ```
# # [Custom algorithms](@id custom_algos)
# 
# All algorithms in ProximalAlgorithms are implemented as iterables.
# The package provides a wrapper, [`IterativeAlgorithm`](@ref), that takes care of looping, stopping and printing, so that no algorithm contains a for-loop.
# See the blog post ["Iterative methods done right (life's too short to write for-loops)"](https://lostella.github.io/blog/iterative-methods-done-right/) for the rationale behind this design.
#
# Each algorithm comes with default functions for `stop` (stopping condition), `solution` (what to return) and `display` (what to print).
# All of these can be overridden, as the examples below show.
# 
# To implement your own method, you need to define:
# 1. an iterator type, holding the problem data and the algorithm parameters;
# 2. a state type, holding the quantities that change at every iteration;
# 3. two `Base.iterate` methods, one returning the initial state and one returning the next state from the current one. Both return `(state, state)`.
#
# You then wrap the iterator in [`IterativeAlgorithm`](@ref).
# The resulting algorithm is called with the problem data as keyword arguments, and returns the solution together with the number of iterations performed.
#
# ```@docs
# ProximalAlgorithms.IterativeAlgorithm
# ```
#
# `IterativeAlgorithm` is not meant to be used directly: define an algorithm-specific constructor on top of it that sets sensible defaults for `stop`, `solution` and `display`, as done in the example below.
#
# ## Example: FISTA with adaptive restarts
# 
# As a worked example, we implement FISTA with adaptive restart, using the gradient-based scheme of [ODonoghue2015](@cite).
# Momentum can make FISTA oscillate: the restart rule resets the momentum whenever it starts pushing against the direction of descent.
# We then apply it to the LASSO problem
#
# ```math
# \text{minimize} \quad \tfrac{1}{2}\|Ax - b\|^2 + \lambda \|x\|_1 .
# ```

using ProximalAlgorithms
using ProximalCore, ProximalOperators
using ProximalCore: Zero, gradient!, prox!
using LinearAlgebra, Printf

# Iterator: problem data and algorithm parameters (immutable).
# `gamma` defaults to `1/Lf`, and `restart` switches the restart rule on or off.
Base.@kwdef struct RestartFISTAIteration{R,Tx,Tf,Tg}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    Lf::R
    gamma::R = 1 / Lf
    restart::Bool = true
end

# State: quantities updated at every iteration (mutable, modified in place).
# `res` is the residual used by the stopping condition, `nrestarts` counts the restarts.
mutable struct RestartFISTAState{R,Tx}
    x::Tx
    x_prev::Tx
    y::Tx
    grad_y::Tx
    z::Tx
    t::R
    res::R
    nrestarts::Int
end

# Initial state: the momentum parameter `t` starts at 1.
function Base.iterate(iter::RestartFISTAIteration)
    x = copy(iter.x0)
    R = real(eltype(x))
    state = RestartFISTAState(x, similar(x), copy(x), similar(x), similar(x), one(R), R(Inf), 0)
    return state, state
end

# Gradient restart test: restart when the momentum points against the direction of descent, i.e. when ``\langle y - x^+, x^+ − x \rangle > 0``.
restart_condition(state::RestartFISTAState) = real(dot(state.y .- state.x, state.x .- state.x_prev)) > 0

# One iteration: gradient step at `y`, proximal step, then either a momentum update or a restart.
function Base.iterate(iter::RestartFISTAIteration, state::RestartFISTAState)
    gradient!(state.grad_y, iter.f, state.y)
    state.z .= state.y .- iter.gamma .* state.grad_y
    state.x_prev .= state.x
    prox!(state.x, iter.g, state.z, iter.gamma)
    state.res = norm(state.x .- state.y) / iter.gamma

    if iter.restart && restart_condition(state)
        state.t = one(state.t)
        state.y .= state.x
        state.nrestarts += 1
    else
        t_new = (1 + sqrt(1 + 4 * state.t^2)) / 2
        state.y .= state.x .+ ((state.t - 1) / t_new) .* (state.x .- state.x_prev)
        state.t = t_new
    end
    return state, state
end

# Constructor with defaults: sets the default `stop`, `solution` and `display` on top of `IterativeAlgorithm`.
# By default it stops when the residual falls below `tol`, returns the last iterate, and prints the residual and number of restarts every `freq` iterations.
RestartFISTA(; maxit=100_000,
    tol=1e-8,
    stop=(iter, state) -> state.res <= tol,
    solution=(iter, state) -> state.x,
    verbose=false,
    freq=100,
    display=(it, iter, state) ->
        @printf("%6d | res=%.2e | restarts=%d\n", it, state.res, state.nrestarts),
    kwargs...) =
    ProximalAlgorithms.IterativeAlgorithm(RestartFISTAIteration;
        maxit, stop, solution, verbose, freq, display, kwargs...)

# Random LASSO problem data.
A, b, λ = randn(100, 300), randn(100), 0.1
Lf = opnorm(A)^2
TOL = 1e-7

# `alg` stores only the algorithm options; the problem data are passed when calling it.
alg = RestartFISTA(tol=TOL, verbose=true, freq=1000)
x, iters = alg(x0=zeros(300), f=LeastSquares(A, b), g=NormL1(λ), Lf=Lf)

# Keyword arguments given at call time are forwarded to the iterator, so restart can be switched off here.
x2, iters2 = alg(x0=zeros(300), f=LeastSquares(A, b), g=NormL1(λ), Lf=Lf, restart=false)

# FastForwardBackward evaluates ``\nabla f`` through `value_and_gradient`, so we define it for [`LeastSquares`](https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.LeastSquares) here.
function ProximalAlgorithms.value_and_gradient(f::LeastSquares, x)
    tmp = f.A * x - f.b
    fx = 0.5*f.lambda*norm(tmp, 2)^2
    dfx = f.A' * tmp
    dfx .*= f.lambda
    return fx, dfx
end

FBalg = ProximalAlgorithms.FastForwardBackward(tol=TOL, verbose=true, freq=1000)
xFB, itersFB = FBalg(x0=zeros(300), f=LeastSquares(A, b), g=NormL1(λ), Lf=Lf)

# ## Example: monitoring convergence with a custom stop
# To monitor convergence, we can pass our own `stop` function.
# It is called after every iteration with the iterator and the new state, and returns `true` when the algorithm should stop.
# Here we record the residual in a vector before applying the usual tolerance test, which gives a lightweight callback without modifying the algorithm.
history = Float64[]
alg = RestartFISTA(tol=TOL, stop=(iter, state) -> begin
    push!(history, state.res)
    state.res <= TOL
end)

# We run FISTA with and without restart, recording the residuals, and plot them.
# For reference, we record the same quantity for the package's own `FastForwardBackward`, which does not restart.
# Its curve coincides with plain FISTA, as expected: the two implement the same algorithm.
using Plots

x, iters = alg(x0=zeros(300), f=LeastSquares(A, b), g=NormL1(λ), Lf=Lf)
plot(history, label="FISTA with restarts", yscale=:log10)

empty!(history)
x2, iters2 = alg(x0=zeros(300), f=LeastSquares(A, b), g=NormL1(λ), Lf=Lf, restart=false)
plot!(history, label="FISTA")

empty!(history)
FBalg = ProximalAlgorithms.FastForwardBackward(tol=TOL, stop=(iter, state) -> begin
    push!(history, norm(state.res) / state.gamma)
    norm(state.res) / state.gamma <= TOL
end)
xFB, itersFB = FBalg(x0=zeros(300), f=LeastSquares(A, b), g=NormL1(λ), Lf=Lf)
plot!(history, label="FastForwardBackward")

# The plot shows the effect of restarting: the residual decreases more quickly when close to a solution and reaches the tolerance in fewer iterations.
# Implementing this variant took an iterator, a state and two `iterate` methods.
# Stopping, solution and printing were customized through `IterativeAlgorithm`, without writing a loop.
