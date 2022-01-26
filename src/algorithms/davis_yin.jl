# Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization
# Applications", Set-Valued and Variational Analysis, vol. 25, no. 4,
# pp. 829–858 (2017).

using Printf
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    DavisYinIteration(; <keyword-arguments>)

Iterator implementing the Davis-Yin splitting algorithm [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x) + h(x),

where `h` is smooth.

See also [`DavisYin`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: proximable objective term.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: smooth objective term.
- `Lh=nothing`: Lipschitz constant of the gradient of h.
- `gamma=nothing`: stepsize to use, defaults to `1/Lh` if not set (but `Lh` is).

# References
1. Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization Applications", Set-Valued and Variational Analysis, vol. 25, no. 4, pp. 829–858 (2017).
"""
Base.@kwdef struct DavisYinIteration{R,C<:Union{R,Complex{R}},T<:AbstractArray{C},Tf,Tg,Th}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    x0::T
    lambda::R = real(eltype(x0))(1)
    Lh::Maybe{R} = nothing
    gamma::Maybe{R} = Lh !== nothing ? (1 / Lh) : error("You must specify either Lh or gamma")
end

Base.IteratorSize(::Type{<:DavisYinIteration}) = Base.IsInfinite()

struct DavisYinState{T}
    z::T
    xg::T
    grad_h_xg::T
    z_half::T
    xf::T
    res::T
end

function Base.iterate(iter::DavisYinIteration)
    z = copy(iter.x0)
    xg, = prox(iter.g, z, iter.gamma)
    grad_h_xg, = gradient(iter.h, xg)
    z_half = 2 .* xg .- z .- iter.gamma .* grad_h_xg
    xf, = prox(iter.f, z_half, iter.gamma)
    res = xf - xg
    z .+= iter.lambda .* res
    state = DavisYinState(z, xg, grad_h_xg, z_half, xf, res)
    return state, state
end

function Base.iterate(iter::DavisYinIteration, state::DavisYinState)
    prox!(state.xg, iter.g, state.z, iter.gamma)
    gradient!(state.grad_h_xg, iter.h, state.xg)
    state.z_half .= 2 .* state.xg .- state.z .- iter.gamma .* state.grad_h_xg
    prox!(state.xf, iter.f, state.z_half, iter.gamma)
    state.res .= state.xf .- state.xg
    state.z .+= iter.lambda .* state.res
    return state, state
end

default_stopping_criterion(tol, ::DavisYinIteration, state::DavisYinState) = norm(state.res, Inf) <= tol
default_solution(::DavisYinIteration, state::DavisYinState) = state.xf
default_display(it, ::DavisYinIteration, state::DavisYinState) = @printf("%5d | %.3e\n", it, norm(state.res, Inf))

"""
    DavisYin(; <keyword-arguments>)

Constructs the Davis-Yin splitting algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x) + h(x),

where `h` is smooth.

The returned object has type `IterativeAlgorithm{DavisYinIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`DavisYinIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs`: keyword arguments to pass on to the `DavisYinIteration` constructor upon call

# References
1. Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization Applications", Set-Valued and Variational Analysis, vol. 25, no. 4, pp. 829–858 (2017).
"""
DavisYin(;
    maxit=10_000,
    tol=1e-8,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...
) = IterativeAlgorithm(DavisYinIteration; maxit, stop, solution, verbose, freq, display, kwargs...)
