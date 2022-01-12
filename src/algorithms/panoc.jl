# Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm
# for nonlinear model predictive control", 56th IEEE Conference on Decision
# and Control (2017).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    PANOCIteration(; <keyword-arguments>)

Instantiate the PANOC algorithm (see [1]) for solving optimization problems
of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).

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

f_model(iter::PANOCIteration, state::PANOCState) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::PANOCIteration{R}) where R
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)
    gamma = iter.gamma === nothing ? iter.alpha / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax) : iter.gamma
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)
    state = PANOCState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, z=z, g_z=g_z, res=x - z, H=initialize(iter.directions, x),
    )
    return state, state
end

set_next_direction!(::QuasiNewtonStyle, ::PANOCIteration, state::PANOCState) = mul!(state.d, state.H, -state.res)
set_next_direction!(::NoAccelerationStyle, ::PANOCIteration, state::PANOCState) = state.d .= .-state.res
set_next_direction!(iter::PANOCIteration, state::PANOCState) = set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

update_direction_state!(::QuasiNewtonStyle, ::PANOCIteration, state::PANOCState) = update!(state.H, state.x - state.x_prev, state.res - state.res_prev)
update_direction_state!(::NoAccelerationStyle, ::PANOCIteration, state::PANOCState) = return
update_direction_state!(iter::PANOCIteration, state::PANOCState) = update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

reset_direction_state!(::QuasiNewtonStyle, ::PANOCIteration, state::PANOCState) = reset!(state.H)
reset_direction_state!(::NoAccelerationStyle, ::PANOCIteration, state::PANOCState) = return
reset_direction_state!(iter::PANOCIteration, state::PANOCState) = reset_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::PANOCIteration{R}, state::PANOCState) where R
    f_Az, a, b, c = R(Inf), R(Inf), R(Inf), R(Inf)

    f_Az_upp = if iter.adaptive == true
        gamma_prev = state.gamma
        state.gamma, state.g_z, f_Az, f_Az_upp = backtrack_stepsize!(
            state.gamma, iter.f, iter.A, iter.g,
            state.x, state.f_Ax, state.At_grad_f_Ax, state.y, state.z, state.g_z, state.res,
            state.Az, state.grad_f_Az,
            alpha = iter.alpha, minimum_gamma = iter.minimum_gamma,
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
    tau = R(1)
    mul!(state.Ad, iter.A, state.d)

    state.x_d .= state.x .+ state.d
    state.Ax_d .= state.Ax .+ state.Ad
    state.f_Ax_d = gradient!(state.grad_f_Ax_d, iter.f, state.Ax_d)
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

    for _ in 1:iter.max_backtracks
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        FBE_x_new = f_model(iter, state) + state.g_z

        if FBE_x_new <= threshold
            # update metric
            update_direction_state!(iter, state)
            state.tau = tau
            return state, state
        end

        if isinf(f_Az)
            mul!(state.Az, iter.A, state.z_curr)
        end

        tau *= 0.5
        state.x .= tau .* state.x_d .+ (1 - tau) .* state.z_curr
        state.Ax .= tau .* state.Ax_d .+ (1 - tau) .* state.Az

        if ProximalOperators.is_quadratic(iter.f)
            # in case f is quadratic, we can compute its value and gradient
            # along a line using interpolation and linear combinations
            # this allows saving operations
            if isinf(f_Az)
                f_Az = gradient!(state.grad_f_Az, iter.f, state.Az)
            end
            if isinf(c)
                mul!(state.At_grad_f_Az, iter.A', state.grad_f_Az)
                c = f_Az
                b = real(dot(state.Ax_d, state.grad_f_Az)) - real(dot(state.Az, state.grad_f_Az))
                a = state.f_Ax_d - b - c
            end
            state.f_Ax = a * tau^2 + b * tau + c
            state.grad_f_Ax .= tau .* state.grad_f_Ax_d .+ (1 - tau) .* state.grad_f_Az
            state.At_grad_f_Ax .= tau .* state.At_grad_f_Ax_d .+ (1 - tau) .* state.At_grad_f_Az
        else
            # otherwise, in the general case where f is only smooth, we compute
            # one gradient and matvec per backtracking step
            state.f_Ax = gradient!(state.grad_f_Ax, iter.f, state.Ax)
            mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)
        end
    end

    @warn "stepsize `tau` became too small ($(tau)), stopping the iterations"
    return nothing
end

# Solver

struct PANOC{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::PANOC)(x0; kwargs...)
    stop(state::PANOCState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it,
        state.gamma,
        norm(state.res, Inf) / state.gamma,
        state.tau,
    )
    iter = PANOCIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.z, num_iters
end

PANOC(; maxit=1_000, tol=1e-8, verbose=false, freq=10, kwargs...) = 
    PANOC(maxit, tol, verbose, freq, kwargs)
