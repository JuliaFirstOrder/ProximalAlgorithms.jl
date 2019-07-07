# Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm
# for nonlinear model predictive control", 56th IEEE Conference on Decision
# and Control (2017).

using Base.Iterators
using ProximalAlgorithms: LBFGS
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

struct PANOC_iterable{R <: Real, C <: Union{R, Complex{R}}, Tx <: AbstractArray{C}, Tf, TA, Tg}
    f::Tf             # smooth term
    A::TA             # matrix/linear operator
    g::Tg             # (possibly) nonsmooth, proximable term
    x0::Tx            # initial point
    alpha::R          # in (0, 1), e.g.: 0.95
    beta::R           # in (0, 1), e.g.: 0.5
    gamma::Maybe{R}   # stepsize parameter of forward and backward steps
    adaptive::Bool    # enforce adaptive stepsize even if L is provided
    memory::Int       # memory parameter for L-BFGS
end

mutable struct PANOC_state{R <: Real, Tx, TAx}
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
    H::LBFGS.LBFGS_buffer{R} # variable metric
    tau::Maybe{R}     # stepsize (can be nothing since the initial state doesn't have it)
    # some additional storage:
    d::Tx
    Ad::TAx
    x_d::Tx
    Ax_d::TAx
    f_Ax_d::R
    grad_f_Ax_d::TAx
    At_grad_f_Ax_d::Tx
    z_curr::Tx
end

PANOC_state(
    x::Tx, Ax::TAx, f_Ax::R, grad_f_Ax, At_grad_f_Ax, gamma::R, y, z, g_z, res, H, tau
) where {R, Tx, TAx} =
    PANOC_state{R, Tx, TAx}(
        x, Ax, f_Ax, grad_f_Ax, At_grad_f_Ax, gamma, y, z, g_z, res, H, tau,
        zero(x), zero(Ax), zero(x), zero(Ax), zero(R), zero(Ax), zero(x), zero(x)
    )

f_model(state::PANOC_state) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

function Base.iterate(iter::PANOC_iterable{R}) where R
    x = iter.x0
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma

    if gamma === nothing
        # compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
        xeps = x .+ R(1)
        grad_f_Axeps, f_Axeps = gradient(iter.f, iter.A*xeps)
        L = norm(iter.A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
        gamma = iter.alpha/L
    end

    # compute initial forward-backward step
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)

    # compute initial fixed-point residual
    res = x - z

    # initialize variable metric
    H = LBFGS.create(x, iter.memory)

    state = PANOC_state(x, Ax, f_Ax, grad_f_Ax, At_grad_f_Ax, gamma, y, z, g_z, res, H, nothing)

    return state, state
end

function Base.iterate(iter::PANOC_iterable{R}, state::PANOC_state{R, Tx, TAx}) where {R, Tx, TAx}
    Az, f_Az, grad_f_Az, At_grad_f_Az = nothing, nothing, nothing, nothing
    a, b, c = nothing, nothing, nothing

    f_Az_upp = f_model(state)

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.gamma === nothing || iter.adaptive == true
        if state.gamma < 1e-7 # TODO: make this a parameter, or dependent on R?
            @warn "parameter `gamma` became too small ($(state.gamma)), stopping the iterations"
            return nothing
        end
        Az = iter.A*state.z
        grad_f_Az, f_Az = gradient(iter.f, Az)
        tol = 10*eps(R)*(1 + abs(f_Az))
        if f_Az <= f_Az_upp + tol break end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        LBFGS.reset!(state.H)
        f_Az_upp = f_model(state)
    end

    # compute FBE
    FBE_x = f_Az_upp + state.g_z

    # update metric
    LBFGS.update!(state.H, state.x, state.res)

    # compute direction
    mul!(state.d, state.H, -state.res)

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

    sigma = iter.beta * (0.5/state.gamma) * (1 - iter.alpha)
    tol = 10*eps(R)*(1 + abs(FBE_x))
    threshold = FBE_x - sigma * norm(state.res)^2 + tol

    for i = 1:10
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
        FBE_x_new = f_model(state) + state.g_z

        if FBE_x_new <= threshold
            state.tau = tau
            return state, state
        end

        if Az === nothing Az = iter.A * state.z_curr end

        tau *= 0.5
        state.x .= tau .* state.x_d .+ (1-tau) .* state.z_curr
        state.Ax .= tau .* state.Ax_d .+ (1-tau) .* Az

        if ProximalOperators.is_quadratic(iter.f)
            # in case f is quadratic, we can compute its value and gradient
            # along a line using interpolation and linear combinations
            # this allows saving operations
            if grad_f_Az === nothing grad_f_Az, f_Az = gradient(iter.f, Az) end
            if At_grad_f_Az === nothing
                At_grad_f_Az = iter.A' * grad_f_Az
                c = f_Az
                b = real(dot(state.Ax_d .- Az, grad_f_Az))
                a = state.f_Ax_d - b - c
            end
            state.f_Ax = a * tau^2 + b * tau + c
            state.grad_f_Ax .= tau .* state.grad_f_Ax_d .+ (1-tau) .* grad_f_Az
            state.At_grad_f_Ax .= tau .* state.At_grad_f_Ax_d .+ (1-tau) .* At_grad_f_Az
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

struct PANOC{R <: Real}
    alpha::R
    beta::R
    gamma::Maybe{R}
    adaptive::Bool
    memory::Int
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int

    function PANOC{R}(; alpha::R=R(0.95), beta::R=R(0.5),
        gamma::Maybe{R}=nothing, adaptive::Bool=false, memory::Int=5,
        maxit::Int=1000, tol::R=R(1e-8), verbose::Bool=false, freq::Int=10
    ) where R
        @assert 0 < alpha < 1
        @assert 0 < beta < 1
        @assert gamma === nothing || gamma > 0
        @assert memory >= 0
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(alpha, beta, gamma, adaptive, memory, maxit, tol, verbose, freq)
    end
end

function (solver::PANOC{R})(
    x0::AbstractArray{C}; f=Zero(), A=I, g=Zero(), L::Maybe{R}=nothing
) where {R, C <: Union{R, Complex{R}}}

    stop(state::PANOC_state) = norm(state.res, Inf)/state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it, state.gamma, norm(state.res, Inf)/state.gamma,
        (state.tau === nothing ? 0.0 : state.tau)
    )

    if solver.gamma === nothing && L !== nothing
        gamma = solver.alpha/L
    elseif solver.gamma !== nothing
        gamma = solver.gamma
    end

    iter = PANOC_iterable(
        f, A, g, x0,
        solver.alpha, solver.beta, solver.gamma, solver.adaptive, solver.memory
    )
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose iter = tee(sample(iter, solver.freq), disp) end

    num_iters, state_final = loop(iter)

    return state_final.z, num_iters

end

# Outer constructors

"""
    PANOC([gamma, adaptive, memory, maxit, tol, verbose, freq, alpha, beta])

Instantiate the PANOC algorithm (see [1]) for solving optimization problems
of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).
If `solver = PANOC(args...)`, then the above problem is solved with

    solver(x0, [f, A, g, L])

Optional keyword arguments:

* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `alpha/L` if not set (but `L` is).
* `adaptive::Bool` (default: `false`), if true, forces the method stepsize to be adaptively adjusted even if `L` is provided (this behaviour is always enforced if `L` is not provided).
* `memory::Integer` (default: `5`), memory parameter for L-BFGS.
* `maxit::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.
* `alpha::Real` (default: `0.95`), stepsize to inverse-Lipschitz-constant ratio; should be in (0, 1).
* `beta::Real` (default: `0.5`), sufficient decrease parameter; should be in (0, 1).

If `gamma` is not specified at construction time, the following keyword
argument can be used to set the stepsize parameter:

* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).

References:

[1] Stella, Themelis, Sopasakis, Patrinos, "A simple and efficient algorithm
for nonlinear model predictive control", 56th IEEE Conference on Decision
and Control (2017).
"""
PANOC(::Type{R}; kwargs...) where R = PANOC{R}(; kwargs...)
PANOC(; kwargs...) = PANOC(Float64; kwargs...)
