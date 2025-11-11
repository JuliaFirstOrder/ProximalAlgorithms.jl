# Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm
# for nonlinear model predictive control", 56th IEEE Conference on Decision
# and Control (2017).

"""
    PANOCIteration(; <keyword-arguments>)

Iterator implementing the PANOC algorithm [1].

This iterator solves optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).

See also: [`PANOC`](@ref).

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
1. Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm for nonlinear model predictive control", 56th IEEE Conference on Decision and Control (2017).
"""
Base.@kwdef struct PANOCIteration{R,Tx,Tf,TA,Tg,TLf,Tgamma,D}
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

Base.IteratorSize(::Type{<:PANOCIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct PANOCState{R,Tx,TAx,TH}
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
    Ad::TAx = similar(Ax)
    x_d::Tx = similar(x)
    Ax_d::TAx = similar(Ax)
    f_Ax_d::R = zero(real(eltype(x)))
    grad_f_Ax_d::TAx = similar(Ax)
    At_grad_f_Ax_d::Tx = similar(x)
    z_curr::Tx = similar(x)
    Az::TAx = similar(Ax)
    grad_f_Az::TAx = similar(Ax)
    At_grad_f_Az::Tx = similar(x)
end

f_model(iter::PANOCIteration, state::PANOCState) =
    f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::PANOCIteration{R}) where {R}
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
    state = PANOCState(
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
    return state, state
end

function set_next_direction!(::QuasiNewtonStyle, ::PANOCIteration, state::PANOCState)
    mul!(state.d, state.H, state.res)
    state.d .*= -1
end
set_next_direction!(::NoAccelerationStyle, ::PANOCIteration, state::PANOCState) =
    state.d .= .-state.res
set_next_direction!(iter::PANOCIteration, state::PANOCState) =
    set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

function update_direction_state!(::QuasiNewtonStyle, ::PANOCIteration, state::PANOCState)
    state.x_prev .= state.x .- state.x_prev
    state.res_prev .= state.res .- state.res_prev
    update!(state.H, state.x_prev, state.res_prev)
end
update_direction_state!(::NoAccelerationStyle, ::PANOCIteration, state::PANOCState) = return
update_direction_state!(iter::PANOCIteration, state::PANOCState) =
    update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

reset_direction_state!(::QuasiNewtonStyle, ::PANOCIteration, state::PANOCState) =
    reset!(state.H)
reset_direction_state!(::NoAccelerationStyle, ::PANOCIteration, state::PANOCState) = return
reset_direction_state!(iter::PANOCIteration, state::PANOCState) =
    reset_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::PANOCIteration{R,Tx,Tf}, state::PANOCState) where {R,Tx,Tf}
    f_Az, a, b, c = R(Inf), R(Inf), R(Inf), R(Inf)

    f_Az_upp = if iter.adaptive == true
        gamma_prev = state.gamma
        state.gamma, state.g_z, f_Az, f_Az_upp = backtrack_stepsize!(
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
        if state.gamma != gamma_prev
            reset_direction_state!(iter, state)
        end
        f_Az_upp
    else
        f_model(iter, state)
    end

    # compute FBE
    FBE_x = f_Az_upp + state.g_z

    # compute direction
    set_next_direction!(iter, state)

    # store iterate and residual for metric update later on
    state.x_prev .= state.x
    state.res_prev .= state.res

    # backtrack tau 1 → 0
    state.tau = R(1)
    mul!(state.Ad, iter.A, state.d)

    state.x_d .= state.x .+ state.d
    state.Ax_d .= state.Ax .+ state.Ad
    state.f_Ax_d, grad_f_Ax_d = value_and_gradient(iter.f, state.Ax_d)
    state.grad_f_Ax_d .= grad_f_Ax_d
    mul!(state.At_grad_f_Ax_d, adjoint(iter.A), state.grad_f_Ax_d)

    copyto!(state.x, state.x_d)
    copyto!(state.Ax, state.Ax_d)
    copyto!(state.grad_f_Ax, state.grad_f_Ax_d)
    copyto!(state.At_grad_f_Ax, state.At_grad_f_Ax_d)
    copyto!(state.z_curr, state.z)
    state.f_Ax = state.f_Ax_d

    sigma = iter.beta * (0.5 / state.gamma) * (1 - iter.alpha)
    tol = 10 * eps(R) * (1 + abs(FBE_x))
    threshold = FBE_x - sigma * norm(state.res)^2 + tol

    state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
    state.res .= state.x .- state.z
    FBE_x_new = f_model(iter, state) + state.g_z

    for k = 1:iter.max_backtracks
        if FBE_x_new <= threshold
            break
        end

        if isinf(f_Az)
            mul!(state.Az, iter.A, state.z_curr)
        end

        state.tau = k >= iter.max_backtracks ? R(0) : state.tau / 2
        state.x .= state.tau .* state.x_d .+ (1 - state.tau) .* state.z_curr
        state.Ax .= state.tau .* state.Ax_d .+ (1 - state.tau) .* state.Az

        if ProximalCore.is_generalized_quadratic(Tf)
            # in case f is quadratic, we can compute its value and gradient
            # along a line using interpolation and linear combinations
            # this allows saving operations
            if isinf(f_Az)
                f_Az, grad_f_Az = value_and_gradient(iter.f, state.Az)
                state.grad_f_Az .= grad_f_Az
            end
            if isinf(c)
                mul!(state.At_grad_f_Az, iter.A', state.grad_f_Az)
                c = f_Az
                b =
                    real(dot(state.Ax_d, state.grad_f_Az)) -
                    real(dot(state.Az, state.grad_f_Az))
                a = state.f_Ax_d - b - c
            end
            state.f_Ax = a * state.tau^2 + b * state.tau + c
            state.grad_f_Ax .=
                state.tau .* state.grad_f_Ax_d .+ (1 - state.tau) .* state.grad_f_Az
            state.At_grad_f_Ax .=
                state.tau .* state.At_grad_f_Ax_d .+ (1 - state.tau) .* state.At_grad_f_Az
        else
            # otherwise, in the general case where f is only smooth, we compute
            # one gradient and matvec per backtracking step
            state.f_Ax, grad_f_Ax = value_and_gradient(iter.f, state.Ax)
            state.grad_f_Ax .= grad_f_Ax
            mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)
        end

        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        FBE_x_new = f_model(iter, state) + state.g_z
    end

    update_direction_state!(iter, state)

    return state, state
end

default_stopping_criterion(tol, ::PANOCIteration, state::PANOCState) =
    norm(state.res, Inf) / state.gamma <= tol
default_solution(::PANOCIteration, state::PANOCState) = state.z
default_iteration_summary(it, ::PANOCIteration, state::PANOCState) =
    ("" => it, "f(Ax)" => state.f_Ax, "g(z)" => state.g_z, "γ" => state.gamma, "‖x - z‖/γ" => norm(state.res, Inf) / state.gamma)

"""
    PANOC(; <keyword-arguments>)

Constructs the PANOC algorithm [1].

This algorithm solves optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).

The returned object has type `IterativeAlgorithm{PANOCIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`PANOCIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=1_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=10`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `PANOCIteration` constructor upon call

# References
1. Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm for nonlinear model predictive control", 56th IEEE Conference on Decision and Control (2017).
"""
PANOC(;
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
    PANOCIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:PANOCIteration}) = AssumptionGroup(
    OperatorTerm(:f => (is_smooth,), :A => (is_linear,)),
    SimpleTerm(:g => (is_proximable,))
)
