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
- `H=LBFGS(x0, 5)`: variable metric to use to compute line-search directions.

# References
- [1] Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm
for nonlinear model predictive control", 56th IEEE Conference on Decision
and Control (2017).
"""

@Base.kwdef struct PANOCIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,TA,Tg,TH}
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

f_model(state::PANOCState) =
    f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

function Base.iterate(iter::PANOCIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma

    if gamma === nothing
        # compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
        xeps = x .+ R(1)
        grad_f_Axeps, f_Axeps = gradient(iter.f, iter.A * xeps)
        L = norm(iter.A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
        gamma = iter.alpha / L
    end

    # compute initial forward-backward step
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)

    # compute initial fixed-point residual
    res = x - z

    state = PANOCState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, z=z, g_z=g_z, res=res, H=iter.H,
    )

    return state, state
end

function Base.iterate(
    iter::PANOCIteration{R},
    state::PANOCState{R,Tx,TAx},
) where {R,Tx,TAx}
    Az, f_Az, grad_f_Az, At_grad_f_Az = nothing, nothing, nothing, nothing
    a, b, c = nothing, nothing, nothing

    f_Az_upp = f_model(state)

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.gamma === nothing || iter.adaptive == true
        if state.gamma < iter.minimum_gamma
            @warn "parameter `gamma` became too small ($(state.gamma)), stopping the iterations"
            return nothing
        end
        Az = iter.A * state.z
        grad_f_Az, f_Az = gradient(iter.f, Az)
        tol = 10 * eps(R) * (1 + abs(f_Az))
        if f_Az <= f_Az_upp + tol
            break
        end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        reset!(state.H)
        f_Az_upp = f_model(state)
    end

    # compute FBE
    FBE_x = f_Az_upp + state.g_z

    # compute direction
    mul!(state.d, state.H, -state.res)

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

    state.x .= state.x_d
    state.Ax .= state.Ax_d
    state.f_Ax = state.f_Ax_d
    state.grad_f_Ax .= state.grad_f_Ax_d
    state.At_grad_f_Ax .= state.At_grad_f_Ax_d

    copyto!(state.z_curr, state.z)

    sigma = iter.beta * (0.5 / state.gamma) * (1 - iter.alpha)
    tol = 10 * eps(R) * (1 + abs(FBE_x))
    threshold = FBE_x - sigma * norm(state.res)^2 + tol

    for _ in 1:iter.max_backtracks
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        FBE_x_new = f_model(state) + state.g_z

        if FBE_x_new <= threshold
            # update metric
            update!(state.H, state.x - state.x_prev, state.res - state.res_prev)
            state.tau = tau
            return state, state
        end

        if Az === nothing
            Az = iter.A * state.z_curr
        end

        tau *= 0.5
        state.x .= tau .* state.x_d .+ (1 - tau) .* state.z_curr
        state.Ax .= tau .* state.Ax_d .+ (1 - tau) .* Az

        if ProximalOperators.is_quadratic(iter.f)
            # in case f is quadratic, we can compute its value and gradient
            # along a line using interpolation and linear combinations
            # this allows saving operations
            if grad_f_Az === nothing
                grad_f_Az, f_Az = gradient(iter.f, Az)
            end
            if At_grad_f_Az === nothing
                At_grad_f_Az = iter.A' * grad_f_Az
                c = f_Az
                b = real(dot(state.Ax_d .- Az, grad_f_Az))
                a = state.f_Ax_d - b - c
            end
            state.f_Ax = a * tau^2 + b * tau + c
            state.grad_f_Ax .= tau .* state.grad_f_Ax_d .+ (1 - tau) .* grad_f_Az
            state.At_grad_f_Ax .= tau .* state.At_grad_f_Ax_d .+ (1 - tau) .* At_grad_f_Az
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
        (state.tau === nothing ? 0.0 : state.tau)
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
