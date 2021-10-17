# Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
# Optimization" (2008).
#
# Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
# for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2,
# no. 1, pp. 183-202 (2009).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    FastForwardBackwardIteration(; <keyword-arguments>)

Instantiate the accelerated forward-backward splitting algorithm (see [1, 2]) for solving
optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).
- `adaptive=false`: forces the method stepsize to be adaptively adjusted.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.

# References
- [1] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
Optimization" (2008).
- [2] Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1,
pp. 183-202 (2009).
"""

Base.@kwdef struct FastForwardBackwardIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    m::R = real(eltype(x0))(0)
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = false
    minimum_gamma::R = real(eltype(x0))(1e-7)
    theta::R = real(eltype(x0))(1)
end

Base.IteratorSize(::Type{<:FastForwardBackwardIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct FastForwardBackwardState{R,Tx}
    x::Tx             # iterate
    f_x::R            # value f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of g at z
    res::Tx           # fixed-point residual at iterate (= z - x)
    theta::R          # acceleration sequence
    z_prev::Tx = copy(x)
end

function Base.iterate(iter::FastForwardBackwardIteration{R}) where {R}
    x = copy(iter.x0)
    grad_f_x, f_x = gradient(iter.f, x)

    gamma = iter.gamma
    if gamma === nothing
        gamma = 1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x)
    end

    y = x - gamma .* grad_f_x
    z, g_z = prox(iter.g, y, gamma)

    state = FastForwardBackwardState(
        gamma=gamma, theta=iter.theta,
        x=x, f_x=f_x, grad_f_x=grad_f_x,
        y=y, z=z, g_z=g_z, res=x - z,
    )

    return state, state
end

function update_theta(prev_theta, prev_gamma, gamma, m=0)
    # Solves for theta: theta^2 / gamma + (theta - 1) * theta'^2 / gamma' - m * theta = 0
    b = prev_theta^2 / prev_gamma - m
    delta = b^2 + 4 * (prev_theta^2) / (prev_gamma * gamma)
    return gamma * (-b + sqrt(delta)) / 2
end

function Base.iterate(iter::FastForwardBackwardIteration{R}, state::FastForwardBackwardState{R,Tx}) where {R,Tx}
    gamma = if iter.gamma === nothing || iter.adaptive == true
        gamma, state.g_z = backtrack_stepsize!(
            state.gamma, iter.f, I, iter.g,
            state.x, state.f_x, state.grad_f_x, state.y, state.z, state.g_z, state.res,
            minimum_gamma = iter.minimum_gamma,
        )
        gamma
    else
        iter.gamma
    end

    theta = update_theta(state.theta, state.gamma, gamma, iter.m)
    beta = gamma * (state.theta) * (1 - state.theta) / (state.gamma * theta + gamma * state.theta ^ 2)
    state.gamma = gamma
    state.theta = theta

    state.x .= state.z .+ beta .* (state.z .- state.z_prev)
    state.z_prev, state.z = state.z, state.z_prev

    state.f_x = gradient!(state.grad_f_x, iter.f, state.x)
    state.y .= state.x .- state.gamma .* state.grad_f_x
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
    state.res .= state.x .- state.z

    return state, state
end

# Solver

struct FastForwardBackward{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::FastForwardBackward)(x0; kwargs...)
    stop(state::FastForwardBackwardState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) =
        @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)
    iter = FastForwardBackwardIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.z, num_iters
end

FastForwardBackward(; maxit=10_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    FastForwardBackward(maxit, tol, verbose, freq, kwargs)

# Aliases

const FastProximalGradientIteration = FastForwardBackwardIteration
const FastProximalGradient = FastForwardBackward
