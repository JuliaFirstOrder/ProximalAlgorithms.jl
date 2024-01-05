# Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
# Optimization" (2008).
#
# Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
# for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2,
# no. 1, pp. 183-202 (2009).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalCore: Zero
using LinearAlgebra
using Printf

"""
    FastForwardBackwardIteration(; <keyword-arguments>)

Iterator implementing the accelerated forward-backward splitting algorithm [1, 2].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

See also: [`FastForwardBackward`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `mf=0`: convexity modulus of `f`.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize, defaults to `1/Lf` if `Lf` is set, and `nothing` otherwise.
- `adaptive=true`: makes `gamma` adaptively adjust during the iterations; this is by default `gamma === nothing`.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `extrapolation_sequence=nothing`: sequence (iterator) of extrapolation coefficients to use for acceleration.

# References
1. Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization" (2008).
2. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).
"""
Base.@kwdef struct FastForwardBackwardIteration{R,Tx,Tf,Tg,TLf,Tgamma,Textr}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    mf::R = real(eltype(x0))(0)
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = gamma === nothing
    minimum_gamma::R = real(eltype(x0))(1e-7)
    extrapolation_sequence::Textr = nothing
end

Base.IteratorSize(::Type{<:FastForwardBackwardIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct FastForwardBackwardState{R,Tx,Textr}
    x::Tx             # iterate
    f_x::R            # value f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of g at z
    res::Tx           # fixed-point residual at iterate (= z - x)
    z_prev::Tx = copy(x)
    extrapolation_sequence::Textr
end

function Base.iterate(iter::FastForwardBackwardIteration)
    x = copy(iter.x0)
    f_x, pb = value_and_pullback(iter.f, x)
    grad_f_x = pb()
    gamma = iter.gamma === nothing ? 1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x) : iter.gamma
    y = x - gamma .* grad_f_x
    z, g_z = prox(iter.g, y, gamma)
    state = FastForwardBackwardState(
        x=x, f_x=f_x, grad_f_x=grad_f_x, gamma=gamma,
        y=y, z=z, g_z=g_z, res=x - z,
        extrapolation_sequence=if iter.extrapolation_sequence !== nothing
            Iterators.Stateful(iter.extrapolation_sequence)
        else
            AdaptiveNesterovSequence(iter.mf)
        end,
    )
    return state, state
end

get_next_extrapolation_coefficient!(state::FastForwardBackwardState{R,Tx,<:Iterators.Stateful}) where {R, Tx} = first(state.extrapolation_sequence)
get_next_extrapolation_coefficient!(state::FastForwardBackwardState{R,Tx,<:AdaptiveNesterovSequence}) where {R, Tx} = next!(state.extrapolation_sequence, state.gamma)

function Base.iterate(iter::FastForwardBackwardIteration{R}, state::FastForwardBackwardState{R,Tx}) where {R,Tx}
    state.gamma = if iter.adaptive == true
        gamma, state.g_z = backtrack_stepsize!(
            state.gamma, iter.f, nothing, iter.g,
            state.x, state.f_x, state.grad_f_x, state.y, state.z, state.g_z, state.res, state.z, nothing,
            minimum_gamma = iter.minimum_gamma,
        )
        gamma
    else
        iter.gamma
    end

    beta = get_next_extrapolation_coefficient!(state)
    state.x .= state.z .+ beta .* (state.z .- state.z_prev)
    state.z_prev, state.z = state.z, state.z_prev

    state.f_x, pb = value_and_pullback(iter.f, state.x)
    state.grad_f_x .= pb()
    state.y .= state.x .- state.gamma .* state.grad_f_x
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
    state.res .= state.x .- state.z

    return state, state
end

default_stopping_criterion(tol, ::FastForwardBackwardIteration, state::FastForwardBackwardState) = norm(state.res, Inf) / state.gamma <= tol
default_solution(::FastForwardBackwardIteration, state::FastForwardBackwardState) = state.z
default_display(it, ::FastForwardBackwardIteration, state::FastForwardBackwardState) = @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)

"""
    FastForwardBackward(; <keyword-arguments>)

Constructs the accelerated forward-backward splitting algorithm [1, 2].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

The returned object has type `IterativeAlgorithm{FastForwardBackwardIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`FastForwardBackwardIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `FastForwardBackwardIteration` constructor upon call

# References
1. Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization" (2008).
2. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).
"""
FastForwardBackward(;
    maxit=10_000,
    tol=1e-8,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...
) = IterativeAlgorithm(FastForwardBackwardIteration; maxit, stop, solution, verbose, freq, display, kwargs...)

# Aliases

const FastProximalGradientIteration = FastForwardBackwardIteration
const FastProximalGradient = FastForwardBackward
