# Themelis, De Marchi, "A linesearch splitting algorithm for structured optimization
# without global smoothness" (2021).

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
- `H=LBFGS(x0, 5)`: variable metric to use to compute line-search directions.

# References
- [1] Themelis, De Marchi, "A linesearch splitting algorithm for structured
optimization without global smoothness", (2021).
"""

Base.@kwdef struct NOLIPIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,TA,Tg,TH}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (alpha / Lf)
    adaptive::Bool = false
    minimum_gamma::R = real(eltype(x0))(1e-7)
    max_backtracks::Int = 20
    H::TH = LBFGS(x0, 5)
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
    tau::Maybe{R} = nothing
    x_prev::Tx = zero(x)
    res_prev::Tx = zero(x)
    d::Tx = zero(x)
    Ad::TAx = zero(Ax)
    x_d::Tx = zero(x)
    Ax_d::TAx = zero(Ax)
    f_Ax_d::R = zero(real(eltype(x)))
    grad_f_Ax_d::TAx = zero(Ax)
    At_grad_f_Ax_d::Tx = zero(x)
    z_curr::Tx = zero(x)
end

f_model(iter::NOLIPIteration, state::NOLIPState) =
    f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::NOLIPIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma
    if gamma === nothing
        gamma = iter.alpha / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax)
    end

    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)
    res = x - z

    if (iter.gamma === nothing || iter.adaptive == true)
        gamma_prev = gamma
        gamma, g_z, Az, f_Az, grad_f_Az, f_Az_upp = backtrack_stepsize!(
            gamma, iter.f, iter.A, iter.g,
            x, f_Ax, At_grad_f_Ax, y, z, g_z, res,
            alpha = iter.alpha, minimum_gamma = iter.minimum_gamma,
        )
    end

    state = NOLIPState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, z=z, g_z=g_z, res=res, H=iter.H,
    )

    return state, state
end

function Base.iterate(
    iter::NOLIPIteration{R},
    state::NOLIPState{R,Tx,TAx},
) where {R,Tx,TAx}
    Az, f_Az, grad_f_Az, At_grad_f_Az = nothing, nothing, nothing, nothing
    a, b, c = nothing, nothing, nothing

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
            mul!(state.d, state.H, -state.res)
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
            Az = iter.A * state.z
            grad_f_Az, f_Az = gradient(iter.f, Az)
            tol = 10 * eps(R) * (1 + abs(f_Az))
            if f_Az > f_Az_upp + tol && state.gamma >= iter.minimum_gamma
                state.gamma /= 2
                if state.gamma < iter.minimum_gamma
                    @warn "stepsize `gamma` became too small ($(state.gamma))"
                end
                can_update_direction = true
                reset!(state.H)
                continue
            end
        end

        FBE_x = f_Az_upp + state.g_z
        if FBE_x <= threshold
            # update metric
            update!(state.H, state.x - state.x_prev, state.res - state.res_prev)
            return state, state
        else
            state.tau *= 0.5
            if tau_backtracks >= iter.max_backtracks
                @warn "parameter `tau` became too small ($(state.tau)), stopping the iterations"
                return nothing
            end
            tau_backtracks += 1
            can_update_direction = false
        end

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
        (state.tau === nothing ? 0.0 : state.tau)
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
