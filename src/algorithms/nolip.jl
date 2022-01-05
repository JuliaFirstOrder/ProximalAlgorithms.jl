# De Marchi, Themelis, "Proximal gradient algorithms under local Lipschitz
# gradient continuity: a convergence and robustness analysis of PANOC",
# arXiv:2112.13000 (2021).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    NOLIPIteration(; <keyword-arguments>)

Instantiate the NOLIP algorithm (see [1]) for solving optimization problems
of the form

    minimize f(Ax) + g(x),

where `f` is locally smooth and `A` is a linear mapping (for example, a matrix).

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
- [1] De Marchi, Themelis, "Proximal gradient algorithms under local Lipschitz
gradient continuity: a convergence and robustness analysis of PANOC",
arXiv:2112.13000 (2021).
"""

Base.@kwdef struct NOLIPIteration{R,Tx,Tf,TA,Tg,TLf,Tgamma,D}
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

Base.IteratorSize(::Type{<:NOLIPIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct NOLIPState{R,Tx,TAx,TH}
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

f_model(iter::NOLIPIteration, state::NOLIPState) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::NOLIPIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)
    gamma = iter.gamma === nothing ? iter.alpha / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax) : iter.gamma
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)
    state = NOLIPState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, z=z, g_z=g_z, res=x-z, H=initialize(iter.directions, x),
    )
    if (iter.gamma === nothing || iter.adaptive == true)
        state.gamma, state.g_z, f_Az, f_Az_upp = backtrack_stepsize!(
            state.gamma, iter.f, iter.A, iter.g,
            state.x, state.f_Ax, state.At_grad_f_Ax, state.y, state.z, state.g_z, state.res,
            state.Az, state.grad_f_Az,
            alpha = iter.alpha, minimum_gamma = iter.minimum_gamma,
        )
    end
    return state, state
end

set_next_direction!(::QuasiNewtonStyle, ::NOLIPIteration, state::NOLIPState) = mul!(state.d, state.H, -state.res)
set_next_direction!(::NoAccelerationStyle, ::NOLIPIteration, state::NOLIPState) = state.d .= .-state.res
set_next_direction!(iter::NOLIPIteration, state::NOLIPState) = set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

update_direction_state!(::QuasiNewtonStyle, ::NOLIPIteration, state::NOLIPState) = update!(state.H, state.x - state.x_prev, state.res - state.res_prev)
update_direction_state!(::NoAccelerationStyle, ::NOLIPIteration, state::NOLIPState) = return
update_direction_state!(iter::NOLIPIteration, state::NOLIPState) = update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

reset_direction_state!(::QuasiNewtonStyle, ::NOLIPIteration, state::NOLIPState) = reset!(state.H)
reset_direction_state!(::NoAccelerationStyle, ::NOLIPIteration, state::NOLIPState) = return
reset_direction_state!(iter::NOLIPIteration, state::NOLIPState) = reset_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::NOLIPIteration{R}, state::NOLIPState) where R
    f_Az, a, b, c = R(Inf), R(Inf), R(Inf), R(Inf)

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
        else
            state.x .= (1 - state.tau) * (state.x_prev .- state.res_prev) + state.tau * (state.x_prev .+ state.d)
        end

        mul!(state.Ax, iter.A, state.x)
        state.f_Ax = gradient!(state.grad_f_Ax, iter.f, state.Ax)
        mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)

        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z

        f_Az_upp = f_model(iter, state)

        if (iter.gamma === nothing || iter.adaptive == true)
            mul!(state.Az, iter.A, state.z)
            f_Az = gradient!(state.grad_f_Az, iter.f, state.Az)
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

        FBE_x_new = f_Az_upp + state.g_z
        if FBE_x_new <= threshold
            # update metric
            update_direction_state!(iter, state)
            return state, state
        end
        state.tau *= 0.5
        if tau_backtracks > iter.max_backtracks
            @warn "stepsize `tau` became too small ($(state.tau))"
            return nothing
        end
        tau_backtracks += 1
        can_update_direction = false

    end

end

# Solver

struct NOLIP{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::NOLIP)(x0; kwargs...)
    stop(state::NOLIPState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it,
        state.gamma,
        norm(state.res, Inf) / state.gamma,
        state.tau,
    )
    iter = NOLIPIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.z, num_iters
end

NOLIP(; maxit=1_000, tol=1e-8, verbose=false, freq=10, kwargs...) =
    NOLIP(maxit, tol, verbose, freq, kwargs)
