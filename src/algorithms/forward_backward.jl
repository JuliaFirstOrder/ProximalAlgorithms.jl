# Lions, Mercier, “Splitting algorithms for the sum of two nonlinear
# operators,” SIAM Journal on Numerical Analysis, vol. 16, pp. 964–979 (1979).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalCore: Zero
using LinearAlgebra
using Printf

"""
    ForwardBackwardIteration(; <keyword-arguments>)

Iterator implementing the forward-backward splitting algorithm [1].

This iterator solves optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

See also: [`ForwardBackward`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).
- `adaptive=false`: forces the method stepsize to be adaptively adjusted.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.

# References
1. Lions, Mercier, “Splitting algorithms for the sum of two nonlinear operators,” SIAM Journal on Numerical Analysis, vol. 16, pp. 964–979 (1979).
"""
Base.@kwdef struct ForwardBackwardIteration{R,Tx,Tf,Tg,TLf,Tgamma}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = gamma === nothing
    minimum_gamma::R = real(eltype(x0))(1e-7)
end

Base.IteratorSize(::Type{<:ForwardBackwardIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct ForwardBackwardState{R,Tx}
    x::Tx             # iterate
    f_x::R            # value of f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of g at z
    res::Tx           # fixed-point residual at iterate (= z - x)
    Az::Tx=similar(x) # TODO not needed
    grad_f_z::Tx=similar(x)
end

function Base.iterate(iter::ForwardBackwardIteration)
    x = copy(iter.x0)
    grad_f_x, f_x = gradient(iter.f, x)
    gamma = iter.gamma === nothing ? 1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x) : iter.gamma
    y = x - gamma .* grad_f_x
    z, g_z = prox(iter.g, y, gamma)
    state = ForwardBackwardState(
        x=x, f_x=f_x, grad_f_x=grad_f_x,
        gamma=gamma, y=y, z=z, g_z=g_z, res=x - z,
    )
    return state, state
end

function Base.iterate(iter::ForwardBackwardIteration{R}, state::ForwardBackwardState{R,Tx}) where {R,Tx}
    if iter.adaptive == true
        state.gamma, state.g_z, state.f_x = backtrack_stepsize!(
            state.gamma, iter.f, nothing, iter.g,
            state.x, state.f_x, state.grad_f_x, state.y, state.z, state.g_z, state.res, state.z, state.grad_f_z,
            minimum_gamma = iter.minimum_gamma,
        )
        state.x, state.z = state.z, state.x
        state.grad_f_x, state.grad_f_z = state.grad_f_z, state.grad_f_x
    else
        state.x, state.z = state.z, state.x
        state.f_x = gradient!(state.grad_f_x, iter.f, state.x)
    end

    state.y .= state.x .- state.gamma .* state.grad_f_x
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)

    state.res .= state.x .- state.z

    return state, state
end

default_stopping_criterion(tol, ::ForwardBackwardIteration, state::ForwardBackwardState) = norm(state.res, Inf) / state.gamma <= tol
default_solution(::ForwardBackwardIteration, state::ForwardBackwardState) = state.z
default_display(it, ::ForwardBackwardIteration, state::ForwardBackwardState) = @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)

"""
    ForwardBackward(; <keyword-arguments>)

Constructs the forward-backward splitting algorithm [1].

This algorithm solves optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

The returned object has type `IterativeAlgorithm{ForwardBackwardIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`ForwardBackwardIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `ForwardBackwardIteration` constructor upon call

# References
1. Lions, Mercier, “Splitting algorithms for the sum of two nonlinear operators,” SIAM Journal on Numerical Analysis, vol. 16, pp. 964–979 (1979).
"""
ForwardBackward(;
    maxit=10_000,
    tol=1e-8,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...
) = IterativeAlgorithm(ForwardBackwardIteration; maxit, stop, solution, verbose, freq, display, kwargs...)

# Aliases

const ProximalGradientIteration = ForwardBackwardIteration
const ProximalGradient = ForwardBackward
