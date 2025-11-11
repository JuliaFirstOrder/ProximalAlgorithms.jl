# Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization
# Applications", Set-Valued and Variational Analysis, vol. 25, no. 4,
# pp. 829–858 (2017).

"""
    DavisYinIteration(; <keyword-arguments>)

Iterator implementing the Davis-Yin splitting algorithm [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x) + h(x),

where `f` is smooth.

See also [`DavisYin`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of h.
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).

# References
1. Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization Applications", Set-Valued and Variational Analysis, vol. 25, no. 4, pp. 829-858 (2017).
"""
Base.@kwdef struct DavisYinIteration{R,C<:Union{R,Complex{R}},T<:AbstractArray{C},Tf,Tg,Th,TLf}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    x0::T
    lambda::R = real(eltype(x0))(1)
    Lf::TLf = nothing
    gamma::R = Lf !== nothing ? (1 / Lf) : error("You must specify either Lf or gamma")
end

Base.IteratorSize(::Type{<:DavisYinIteration}) = Base.IsInfinite()

struct DavisYinState{T,R}
    z::T
    xg::T
    f_xg::R
    grad_f_xg::T
    z_half::T
    xh::T
    g_xh::R
    res::T
end

function Base.iterate(iter::DavisYinIteration)
    z = copy(iter.x0)
    xg, = prox(iter.g, z, iter.gamma)
    f_xg, grad_f_xg = value_and_gradient(iter.f, xg)
    z_half = 2 .* xg .- z .- iter.gamma .* grad_f_xg
    xh, g_xh = prox(iter.h, z_half, iter.gamma)
    res = xh - xg
    z .+= iter.lambda .* res
    state = DavisYinState(z, xg, f_xg, grad_f_xg, z_half, xh, g_xh, res)
    return state, state
end

function Base.iterate(iter::DavisYinIteration, state::DavisYinState)
    prox!(state.xg, iter.g, state.z, iter.gamma)
    f_xg, grad_f_xg = value_and_gradient(iter.f, state.xg)
    state.grad_f_xg .= grad_f_xg
    state.z_half .= 2 .* state.xg .- state.z .- iter.gamma .* state.grad_f_xg
    prox!(state.xh, iter.h, state.z_half, iter.gamma)
    state.res .= state.xh .- state.xg
    state.z .+= iter.lambda .* state.res
    return state, state
end

default_stopping_criterion(tol, ::DavisYinIteration, state::DavisYinState) =
    norm(state.res, Inf) <= tol
default_solution(::DavisYinIteration, state::DavisYinState) = state.xh
default_iteration_summary(it, ::DavisYinIteration, state::DavisYinState) =
    ("" => it, "f(xg)" => state.f_xg, "g(xh)" => state.g_xh, "‖xg - xh‖" => norm(state.res, Inf))

"""
    DavisYin(; <keyword-arguments>)

Constructs the Davis-Yin splitting algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x) + h(x),

where `f` is smooth.

The returned object has type `IterativeAlgorithm{DavisYinIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`DavisYinIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `DavisYinIteration` constructor upon call

# References
1. Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization Applications", Set-Valued and Variational Analysis, vol. 25, no. 4, pp. 829–858 (2017).
"""
DavisYin(;
    maxit = 10_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
	summary=default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    DavisYinIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:DavisYinIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_convex)),
    SimpleTerm(:g => (is_proximable, is_convex,)),
    SimpleTerm(:h => (is_proximable, is_convex,))
)
