# De Marchi, Themelis, "Proximal Gradient Algorithms under Local Lipschitz
# Gradient Continuity", Journal of Optimization Theory and Applications, 
# vol. 194, no. 3, pp. 771-794 (2022).

"""
    PANOCplusIteration(; <keyword-arguments>)

Iterator implementing the PANOCplus algorithm [1].

This iterator solves optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is locally smooth and `A` is a linear mapping (for example, a matrix).

See also: [`PANOCplus`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `A=I`: linear operator (e.g. a matrix).
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of x ↦ f(Ax).
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).
- `adaptive=false`: forces the method stepsize to be adaptively adjusted.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `max_backtracks=20`: maximum number of line-search backtracks.
- `directions=LBFGS(5)`: strategy to use to compute line-search directions.

# References
1. De Marchi, Themelis, "Proximal Gradient Algorithms under Local Lipschitz Gradient Continuity", Journal of Optimization Theory and Applications, vol. 194, no. 3, pp. 771-794 (2022).
"""
Base.@kwdef struct PANOCplusIteration{R,Tx,Tf,TA,Tg,TLf,Tgamma,D}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (alpha / Lf)
    adaptive::Bool = gamma === nothing
    minimum_gamma::R = real(eltype(x0))(1e-7)
    max_backtracks::Int = 20
    directions::D = LBFGS(5)
end

Base.IteratorSize(::Type{<:PANOCplusIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct PANOCplusState{R,Tx,TAx,TH}
    x::Tx             # iterate
    Ax::TAx           # A times x
    f_Ax::R           # value of smooth term
    grad_f_Ax::TAx    # gradient of f at Ax
    At_grad_f_Ax::Tx  # gradient of smooth term
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of nonsmooth term (at z)
    res::Tx           # fixed-point residual at iterate (= x - z)
    H::TH             # variable metric
    tau::R = zero(gamma)
    x_prev::Tx = similar(x)
    res_prev::Tx = similar(x)
    d::Tx = similar(x)
    Az::TAx = similar(Ax)
    grad_f_Az::TAx = similar(Ax)
    At_grad_f_Az::Tx = similar(x)
end

f_model(iter::PANOCplusIteration, state::PANOCplusState) =
    f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::PANOCplusIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
     f_Ax, grad_f_Ax = value_and_gradient(iter.f, Ax)
    gamma =
        iter.gamma === nothing ?
        iter.alpha / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax) :
        iter.gamma
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)
    state = PANOCplusState(
        x = x,
        Ax = Ax,
        f_Ax = f_Ax,
        grad_f_Ax = grad_f_Ax,
        At_grad_f_Ax = At_grad_f_Ax,
        gamma = gamma,
        y = y,
        z = z,
        g_z = g_z,
        res = x - z,
        H = initialize(iter.directions, x),
    )
    if (iter.gamma === nothing || iter.adaptive == true)
        state.gamma, state.g_z, _, _ = backtrack_stepsize!(
            state.gamma,
            iter.f,
            iter.A,
            iter.g,
            state.x,
            state.f_Ax,
            state.At_grad_f_Ax,
            state.y,
            state.z,
            state.g_z,
            state.res,
            state.Az,
            state.grad_f_Az,
            alpha = iter.alpha,
            minimum_gamma = iter.minimum_gamma,
        )
    else
        mul!(state.Az, iter.A, state.z)
        f_Az, grad_f_Az = value_and_gradient(iter.f, state.Az)
        state.grad_f_Az = grad_f_Az
    end
    mul!(state.At_grad_f_Az, adjoint(iter.A), state.grad_f_Az)
    return state, state
end

function set_next_direction!(
    ::QuasiNewtonStyle,
    ::PANOCplusIteration,
    state::PANOCplusState,
)
    mul!(state.d, state.H, state.res_prev)
    state.d .*= -1
end
set_next_direction!(::NoAccelerationStyle, ::PANOCplusIteration, state::PANOCplusState) =
    state.d .= .-state.res_prev
set_next_direction!(iter::PANOCplusIteration, state::PANOCplusState) =
    set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

function update_direction_state!(
    ::QuasiNewtonStyle,
    ::PANOCplusIteration,
    state::PANOCplusState,
)
    state.x_prev .= state.x .- state.x_prev
    state.res_prev .= state.res .- state.res_prev
    update!(state.H, state.x_prev, state.res_prev)
end
update_direction_state!(
    ::NoAccelerationStyle,
    ::PANOCplusIteration,
    state::PANOCplusState,
) = return
update_direction_state!(iter::PANOCplusIteration, state::PANOCplusState) =
    update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

reset_direction_state!(::QuasiNewtonStyle, ::PANOCplusIteration, state::PANOCplusState) =
    reset!(state.H)
reset_direction_state!(::NoAccelerationStyle, ::PANOCplusIteration, state::PANOCplusState) =
    return
reset_direction_state!(iter::PANOCplusIteration, state::PANOCplusState) =
    reset_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::PANOCplusIteration{R}, state::PANOCplusState) where {R}
    # store iterate and residual for metric update later on
    state.x_prev .= state.x
    state.res_prev .= state.res

    # compute FBE
    FBE_x = f_model(iter, state) + state.g_z

    sigma = iter.beta * (0.5 / state.gamma) * (1 - iter.alpha)
    tol = 10 * eps(R) * (1 + abs(FBE_x))
    threshold = FBE_x - sigma * norm(state.res)^2 + tol

    tau_backtracks = 0
    can_update_direction = true

    while true

        if can_update_direction
            # compute direction
            set_next_direction!(iter, state)
            # backtrack tau 1 → 0
            state.tau = R(1)
            state.x .= state.x_prev .+ state.d
            tau_backtracks = 0
        else
            state.x .=
                (1 - state.tau) .* (state.x_prev .- state.res_prev) .+
                state.tau .* (state.x_prev .+ state.d)
            tau_backtracks += 1
        end

        mul!(state.Ax, iter.A, state.x)
        state.f_Ax, grad_f_Ax = value_and_gradient(iter.f, state.Ax)
        state.grad_f_Ax .= grad_f_Ax
        mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)

        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z

        f_Az_upp = f_model(iter, state)

        mul!(state.Az, iter.A, state.z)
        f_Az, grad_f_Az = value_and_gradient(iter.f, state.Az)
        state.grad_f_Az .= grad_f_Az
        if (iter.gamma === nothing || iter.adaptive == true)
            tol = 10 * eps(R) * (1 + abs(f_Az))
            if f_Az > f_Az_upp + tol && state.gamma >= iter.minimum_gamma
                state.gamma *= 0.5
                if state.gamma < iter.minimum_gamma
                    @warn "stepsize `gamma` became too small ($(state.gamma))"
                end
                can_update_direction = true
                reset_direction_state!(iter, state)
                continue
            end
        end
        mul!(state.At_grad_f_Az, adjoint(iter.A), state.grad_f_Az)

        FBE_x_new = f_Az_upp + state.g_z
        if FBE_x_new <= threshold || tau_backtracks >= iter.max_backtracks
            break
        end
        state.tau = tau_backtracks >= iter.max_backtracks - 1 ? R(0) : state.tau / 2
        can_update_direction = false

    end

    update_direction_state!(iter, state)

    return state, state

end

default_stopping_criterion(tol, ::PANOCplusIteration, state::PANOCplusState) =
    norm((state.res / state.gamma) - state.At_grad_f_Ax + state.At_grad_f_Az, Inf) <= tol
default_solution(::PANOCplusIteration, state::PANOCplusState) = state.z
default_iteration_summary(it, ::PANOCplusIteration, state::PANOCplusState) =
    ("" => it, "f(Ax)" => state.f_Ax, "g(z)" => state.g_z, "γ" => state.gamma, "‖x - z‖/γ" => norm(state.res, Inf) / state.gamma)

"""
    PANOCplus(; <keyword-arguments>)

Constructs the the PANOCplus algorithm [1].

This algorithm solves optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is locally smooth and `A` is a linear mapping (for example, a matrix).

The returned object has type `IterativeAlgorithm{PANOCplusIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`PANOCplusIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=10`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `PANOCplusIteration` constructor upon call

# References
1. De Marchi, Themelis, "Proximal Gradient Algorithms under Local Lipschitz Gradient Continuity", Journal of Optimization Theory and Applications, vol. 194, no. 3, pp. 771-794 (2022).
"""
PANOCplus(;
    maxit = 1_000,
    tol = 1e-8,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 10,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    PANOCplusIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:PANOCplusIteration}) = AssumptionGroup(
    OperatorTerm(:f => (is_smooth,), :A => (is_linear,)),
    SimpleTerm(:g => (is_proximable,))
)
