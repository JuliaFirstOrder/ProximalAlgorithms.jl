# Latafat, Themelis, Stella, Patrinos, "Adaptive proximal algorithms for convex optimization
# under local Lipschitz continuity of the gradient," Mathematical Programming, vol. 213, pp. 433–471 (2025).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalCore: Zero
using LinearAlgebra
using Printf

"""
    AdaptiveProximalGradientIteration(; <keyword-arguments>)

Iterator implementing the adaptive proximal-gradient method [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

See also: [`AdaptiveProximalGradient`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize, defaults to `1/Lf` if `Lf` is set, and `nothing` otherwise.

# References
1. Latafat, Themelis, Stella, Patrinos, "Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient" (2025).
"""
Base.@kwdef struct AdaptiveProximalGradientIteration{Tx,Tf,Tg,TLf,Tgamma}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (1 / Lf)
end

Base.IteratorSize(::Type{<:AdaptiveProximalGradientIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct AdaptiveProximalGradientState{R,Tx}
    x::Tx             # iterate
    f_x::R            # value f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    f_z::R            # value f at z
    g_z::R            # value of g at z
    grad_f_z::Tx      # gradient of f at z
    res::Tx           # fixed-point residual at iterate
    gamma_old::R      # previous stepsize parameter
end

function Base.iterate(iter::AdaptiveProximalGradientIteration)
    x = copy(iter.x0)
    f_x, grad_f_x = value_and_gradient(iter.f, x)
    gamma =
        iter.gamma === nothing ?
        1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x) : iter.gamma
    y = x - gamma .* grad_f_x
    z, g_z = prox(iter.g, y, gamma)
    f_z, grad_f_z = value_and_gradient(iter.f, z)
    state = AdaptiveProximalGradientState(
        x=x,
        f_x=f_x,
        grad_f_x=grad_f_x,
        gamma=gamma,
        y=y,
        z=z,
        f_z=f_z,
        g_z=g_z,
        grad_f_z=grad_f_z,
        res=x - z,
        gamma_old=gamma,
    )
    return state, state
end

nan_to_zero(v) = ifelse(isnan(v), zero(v), v)

# stepsize rule as defined in Algorithm 2.1 of [1]
function get_stepsize(gamma, x, grad_x, gamma_old, x_old, grad_x_old)
    C = norm(grad_x - grad_x_old)^2 / real(dot(grad_x - grad_x_old, x - x_old)) |> nan_to_zero
    L = real(dot(grad_x - grad_x_old, x - x_old)) / norm(x - x_old)^2 |> nan_to_zero
    D = max(0, gamma * L * (gamma * C - 1))
    gamma_new = gamma * min(sqrt(1 + gamma / gamma_old), 1 / (2 * sqrt(D)))
    return gamma_new, gamma
end

function Base.iterate(
    iter::AdaptiveProximalGradientIteration,
    state::AdaptiveProximalGradientState{R,Tx},
) where {R,Tx}
    # estimate stepsize
    state.gamma, state.gamma_old = get_stepsize(state.gamma, state.z, state.grad_f_z, state.gamma_old, state.x, state.grad_f_x)
    # shift
    state.x .= state.z
    state.grad_f_x .= state.grad_f_z
    # compute new point
    state.y .= state.x .- state.gamma .* state.grad_f_x
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
    state.f_z, state.grad_f_z = value_and_gradient(iter.f, state.z)
    state.res .= state.x .- state.z
    return state, state
end

default_stopping_criterion(
    tol,
    ::AdaptiveProximalGradientIteration,
    state::AdaptiveProximalGradientState,
) = norm(state.res, Inf) / state.gamma <= tol
default_solution(::AdaptiveProximalGradientIteration, state::AdaptiveProximalGradientState) = state.z
default_display(it, ::AdaptiveProximalGradientIteration, state::AdaptiveProximalGradientState) =
    @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)

"""
    AdaptiveProximalGradient(; <keyword-arguments>)

Constructs the adaptive proximal-gradient algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is locally smooth.

The returned object has type `IterativeAlgorithm{AdaptiveProximalGradientIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`AdaptiveProximalGradientIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AdaptiveProximalGradientIteration` constructor upon call

# References
1. Latafat, Themelis, Stella, Patrinos, "Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient" (2025).
"""
AdaptiveProximalGradient(;
    maxit=10_000,
    tol=1e-8,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...,
) = IterativeAlgorithm(
    AdaptiveProximalGradientIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    display,
    kwargs...,
)

# Aliases

const AdaPGMState = AdaptiveProximalGradientState
const AdaPGMIteration = AdaptiveProximalGradientIteration
const AdaPGM = AdaptiveProximalGradient