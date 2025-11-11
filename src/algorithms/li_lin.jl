# Li, Lin, "Accelerated Proximal Gradient Methods for Nonconvex Programming",
# Proceedings of NIPS 2015 (2015).

"""
    LiLinIteration(; <keyword-arguments>)

Iterator implementing the nonconvex accelerated proximal gradient method by Li and Lin
(see Algorithm 2 in [1]).

This iterator solves optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

See also: [`LiLin`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of f.
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).

# References
1. Li, Lin, "Accelerated Proximal Gradient Methods for Nonconvex Programming", Proceedings of NIPS 2015 (2015).
"""
Base.@kwdef struct LiLinIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = false
    delta::R = real(eltype(x0))(1e-3)
    eta::R = real(eltype(x0))(0.8)
end

Base.IteratorSize(::Type{<:LiLinIteration}) = Base.IsInfinite()

mutable struct LiLinState{R<:Real,Tx}
    x::Tx             # iterate
    y::Tx             # extrapolated point
    f_y::R            # value of f at y
    grad_f_y::Tx      # gradient of f at y
    # TODO: *two* gammas should be used in general, one for y and one for x
    gamma::R          # stepsize parameter of forward and backward steps
    y_forward::Tx     # forward point at y
    z::Tx             # forward-backward point
    g_z::R            # value of nonsmooth term at z
    res::Tx           # fixed-point-residual (at y)
    theta::R          # auxiliary sequence to compute extrapolated points
    F_average::R      # moving average of objective values
    q::R              # auxiliary sequence to compute moving average
end

function Base.iterate(iter::LiLinIteration{R}) where {R}
    y = copy(iter.x0)
    f_y, grad_f_y = value_and_gradient(iter.f, y)

    # TODO: initialize gamma if not provided
    # TODO: authors suggest Barzilai-Borwein rule?
    # TODO: *two* gammas should be used in general, one for y and one for x

    # compute initial forward-backward step
    y_forward = y - iter.gamma .* grad_f_y
    z, g_z = prox(iter.g, y_forward, iter.gamma)

    Fy = f_y + iter.g(y)

    @assert isfinite(Fy) "initial point must be feasible"

    # compute initial fixed-point residual
    res = y - z

    state = LiLinState(
        copy(iter.x0),
        y,
        f_y,
        grad_f_y,
        iter.gamma,
        y_forward,
        z,
        g_z,
        res,
        R(1),
        Fy,
        R(1),
    )

    return state, state
end

function Base.iterate(iter::LiLinIteration{R}, state::LiLinState{R,Tx}) where {R,Tx}
    # TODO: backtrack gamma at y

    Fz = iter.f(state.z) + state.g_z

    theta1 = (R(1) + sqrt(R(1) + 4 * state.theta^2)) / R(2)

    if Fz <= state.F_average - iter.delta * norm(state.res)^2
        case = 1
    else
        # TODO: re-use available space in state?
        # TODO: backtrack gamma at x
        f_x, grad_f_x = value_and_gradient(iter.f, x)
        x_forward = state.x - state.gamma .* grad_f_x
        v, g_v = prox(iter.g, x_forward, state.gamma)
        Fv = iter.f(v) + g_v
        case = Fz <= Fv ? 1 : 2
    end

    if case == 1
        state.y .= state.z .+ ((state.theta - R(1)) / theta1) .* (state.z .- state.x)
        state.x, state.z = state.z, state.x
        Fx = Fz
    elseif case == 2
        state.y .=
            state.z .+ (state.theta / theta1) .* (state.z .- v) .+
            ((state.theta - R(1)) / theta1) .* (v .- state.x)
        state.x = v
        Fx = Fv
    end

    state.f_y, grad_f_y = value_and_gradient(iter.f, state.y)
    state.grad_f_y .= grad_f_y
    state.y_forward .= state.y .- state.gamma .* state.grad_f_y
    state.g_z = prox!(state.z, iter.g, state.y_forward, state.gamma)

    state.res .= state.y - state.z

    state.theta = theta1

    # NOTE: the following can be simplified
    q1 = iter.eta * state.q + 1
    state.F_average = (iter.eta * state.q * state.F_average + Fx) / q1
    state.q = q1

    return state, state
end

default_stopping_criterion(tol, ::LiLinIteration, state::LiLinState) =
    norm(state.res, Inf) / state.gamma <= tol
default_solution(::LiLinIteration, state::LiLinState) = state.z
default_iteration_summary(it, ::LiLinIteration, state::LiLinState) =
    ("" => it, "γ" => state.gamma, "f(y)" => state.f_y, "g(z)" => state.g_z, "‖y - z‖/γ" => norm(state.res, Inf) / state.gamma)

"""
    LiLin(; <keyword-arguments>)

Constructs the nonconvex accelerated proximal gradient method by Li and Lin
(see Algorithm 2 in [1]).

This algorithm solves optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

The returned object has type `IterativeAlgorithm{LiLinIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`LiLinIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `LiLinIteration` constructor upon call

# References
1. Li, Lin, "Accelerated Proximal Gradient Methods for Nonconvex Programming", Proceedings of NIPS 2015 (2015).
"""
LiLin(;
    maxit = 10_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    LiLinIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:LiLinIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth,)),
    SimpleTerm(:g => (is_proximable,))
)
