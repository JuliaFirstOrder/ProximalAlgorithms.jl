# Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).

"""
    DouglasRachfordIteration(; <keyword-arguments>)

Iterator implementing the Douglas-Rachford splitting algorithm [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x).

See also: [`DouglasRachford`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: proximable objective term.
- `g=Zero()`: proximable objective term.
- `gamma`: stepsize to use.

# References
1. Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators", Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
"""
Base.@kwdef struct DouglasRachfordIteration{
    R,
    C<:Union{R,Complex{R}},
    Tx<:AbstractArray{C},
    Tf,
    Tg,
}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    gamma::R
end

Base.IteratorSize(::Type{<:DouglasRachfordIteration}) = Base.IsInfinite()

Base.@kwdef struct DouglasRachfordState{Tx,R}
    x::Tx
    y::Tx = similar(x)
    f_y::R = real(eltype(x))(0)
    r::Tx = similar(x)
    z::Tx = similar(x)
    g_z::R = real(eltype(x))(0)
    res::Tx = similar(x)
end

function Base.iterate(
    iter::DouglasRachfordIteration,
    state::DouglasRachfordState = DouglasRachfordState(x = copy(iter.x0)),
)
    f_y = prox!(state.y, iter.f, state.x, iter.gamma)
    state.r .= 2 .* state.y .- state.x
    g_z = prox!(state.z, iter.g, state.r, iter.gamma)
    state.res .= state.y .- state.z
    state.x .-= state.res
    state = DouglasRachfordState(state.x, state.y, f_y, state.r, state.z, g_z, state.res)
    return state, state
end

default_stopping_criterion(
    tol,
    iter::DouglasRachfordIteration,
    state::DouglasRachfordState,
) = norm(state.res, Inf) / iter.gamma <= tol
default_solution(::DouglasRachfordIteration, state::DouglasRachfordState) = state.y
default_iteration_summary(it, iter::DouglasRachfordIteration, state::DouglasRachfordState) =
    ("" => it, "f(y)" => state.f_y, "g(z)" => state.g_z, "‖y - z‖" => norm(state.res, Inf) / iter.gamma)

"""
    DouglasRachford(; <keyword-arguments>)

Constructs the Douglas-Rachford splitting algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x).

The returned object has type `IterativeAlgorithm{DouglasRachfordIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`DouglasRachfordIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `DouglasRachfordIteration` constructor upon call

# References
1. Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators", Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
"""
DouglasRachford(;
    maxit = 1_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    DouglasRachfordIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:DouglasRachfordIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_proximable,)),
    SimpleTerm(:g => (is_proximable,))
)
