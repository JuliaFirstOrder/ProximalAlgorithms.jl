# Li, Lin, "Accelerated Proximal Gradient Methods for Nonconvex Programming",
# Proceedings of NIPS 2015 (2015).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

struct LiLin_iterable{R <: Real, C <: Union{R, Complex{R}}, Tx <: AbstractArray{C}, Tf, TA, Tg}
    f::Tf             # smooth term
    A::TA             # matrix/linear operator
    g::Tg             # (possibly) nonsmooth, proximable term
    x0::Tx            # initial point
    gamma::Maybe{R}   # stepsize parameter of forward and backward steps
    adaptive::Bool    # enforce adaptive stepsize even if L is provided
    delta::R          #
    eta::R            #
end

mutable struct LiLin_state{R <: Real, Tx, TAx}
    x::Tx             # iterate
    y::Tx             # extrapolated point
    Ay::TAx           # A times y
    f_Ay::R           # value of smooth term at y
    grad_f_Ay::TAx    # gradient of f at Ay
    At_grad_f_Ay::Tx  # gradient of smooth term at y
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

function Base.iterate(iter::LiLin_iterable{R}) where R
    y = iter.x0
    Ay = iter.A * y
    grad_f_Ay, f_Ay = gradient(iter.f, Ay)

    # TODO: initialize gamma if not provided
    # TODO: authors suggest Barzilai-Borwein rule?
    # TODO: *two* gammas should be used in general, one for y and one for x

    # compute initial forward-backward step
    At_grad_f_Ay = iter.A' * grad_f_Ay
    y_forward = y - iter.gamma .* At_grad_f_Ay
    z, g_z = prox(iter.g, y_forward, iter.gamma)

    Fy = f_Ay + iter.g(y)

    @assert isfinite(Fy) "initial point must be feasible"

    # compute initial fixed-point residual
    res = y - z

    state = LiLin_state(
        copy(iter.x0), y, Ay, f_Ay, grad_f_Ay, At_grad_f_Ay,
        iter.gamma, y_forward, z, g_z, res, R(1), Fy, R(1)
    )

    return state, state
end

function Base.iterate(iter::LiLin_iterable{R}, state::LiLin_state{R, Tx, TAx}) where {R, Tx, TAx}
    # TODO: backtrack gamma at y

    Fz = iter.f(state.z) + state.g_z

    theta1 = (R(1)+sqrt(R(1)+4*state.theta^2))/R(2)

    if Fz <= state.F_average - iter.delta * norm(state.res)^2
        case = 1
    else
        # TODO: re-use available space in state?
        # TODO: backtrack gamma at x
        Ax = iter.A * state.x
        grad_f_Ax, f_Ax = gradient(iter.f, Ax)
        At_grad_f_Ax = iter.A' * grad_f_Ax
        x_forward = state.x - state.gamma .* At_grad_f_Ax
        v, g_v = prox(iter.g, x_forward, state.gamma)
        Fv = iter.f(v) + g_v
        case = Fz <= Fv ? 1 : 2
    end

    if case == 1
        state.y .= state.z .+ ((state.theta - R(1)) / theta1) .* (state.z .- state.x)
        state.x, state.z = state.z, state.x
        Fx = Fz
    elseif case == 2
        state.y .= state.z .+ (state.theta / theta1) .* (state.z .- v) .+ ((state.theta - R(1)) / theta1) .* (v .- state.x)
        state.x = v
        Fx = Fv
    end

    mul!(state.Ay, iter.A, state.y)
    state.f_Ay = gradient!(state.grad_f_Ay, iter.f, state.Ay)
    mul!(state.At_grad_f_Ay, adjoint(iter.A), state.grad_f_Ay)
    state.y_forward .= state.y .- state.gamma .* state.At_grad_f_Ay
    state.g_z = prox!(state.z, iter.g, state.y_forward, state.gamma)

    state.res .= state.y - state.z

    state.theta = theta1

    # NOTE: the following can be simplified
    q1 = iter.eta * state.q + 1
    state.F_average = (iter.eta * state.q * state.F_average + Fx)/q1
    state.q = q1

    return state, state
end

# Solver

struct LiLin{R <: Real}
    gamma::Maybe{R}
    adaptive::Bool
    delta::R
    eta::R
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int

    function LiLin{R}(; gamma::Maybe{R}=nothing, adaptive::Bool=false,
        delta::R=R(1e-3), eta::R=R(0.8), maxit::Int=10000, tol::R=R(1e-8),
        verbose::Bool=false, freq::Int=100
    ) where R
        @assert gamma === nothing || gamma > 0
        @assert delta > 0
        @assert 0 < eta < 1
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(gamma, adaptive, delta, eta, maxit, tol, verbose, freq)
    end
end

function (solver::LiLin{R})(
    x0::AbstractArray{C}; f=Zero(), A=I, g=Zero(), L::Maybe{R}=nothing
) where {R, C <: Union{R, Complex{R}}}

    stop(state::LiLin_state) = norm(state.res, Inf)/state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e\n",
        it, state.gamma, norm(state.res, Inf)/state.gamma
    )

    if solver.gamma === nothing && L !== nothing
        gamma = R(1)/L
    elseif solver.gamma !== nothing
        gamma = solver.gamma
    else
        gamma = nothing
    end

    iter = LiLin_iterable(f, A, g, x0, gamma, solver.adaptive, solver.delta, solver.eta)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose iter = tee(sample(iter, solver.freq), disp) end

    num_iters, state_final = loop(iter)

    return state_final.z, num_iters

end

# Outer constructors

"""
    LiLin([gamma, adaptive, fast, maxit, tol, verbose, freq])

Instantiate the nonconvex accelerated proximal gradient method by Li and Lin
(see Algorithm 2 in [1]) for solving optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).
If `solver = LiLin(args...)`, then the above problem is solved with

    solver(x0, [f, A, g, L])

Optional keyword arguments:

* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `1/L` if not set (but `L` is).
* `adaptive::Bool` (default: `false`), if true, forces the method stepsize to be adaptively adjusted.
* `delta::Real` (default: `1e-3`), parameter determinining when extrapolated steps are to be accepted.
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.

If `gamma` is not specified at construction time, the following keyword
argument can be used to set the stepsize parameter:

* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).

References:

[1] Li, Lin, "Accelerated Proximal Gradient Methods for Nonconvex Programming",
Proceedings of NIPS 2015 (2015).
"""
LiLin(::Type{R}; kwargs...) where R = LiLin{R}(; kwargs...)
LiLin(; kwargs...) = LiLin(Float64; kwargs...)
