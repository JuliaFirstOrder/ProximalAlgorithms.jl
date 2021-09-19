# Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
# Optimization" (2008).
#
# Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
# for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2,
# no. 1, pp. 183-202 (2009).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    FastForwardBackwardIteration(; <keyword-arguments>)

Instantiate the accelerated forward-backward splitting algorithm (see [1, 2]) for solving
optimization problems of the form

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

# References
- [1] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
Optimization" (2008).
- [2] Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1,
pp. 183-202 (2009).
"""

Base.@kwdef struct FastForwardBackwardIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,TA,Tg}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = false
    minimum_gamma::R = real(eltype(x0))(1e-7)
end

Base.IteratorSize(::Type{<:FastForwardBackwardIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct FastForwardBackwardState{R,Tx,TAx}
    x::Tx             # iterate
    Ax::TAx           # A times x
    f_Ax::R           # value of smooth term
    grad_f_Ax::TAx    # gradient of f at Ax
    At_grad_f_Ax::Tx  # gradient of smooth term
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of nonsmooth term (at z)
    res::Tx           # fixed-point residual at iterate (= z - x)
    theta::R = one(real(eltype(x)))
    z_prev::Tx = copy(x)
end

f_model(state::FastForwardBackwardState) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

function Base.iterate(iter::FastForwardBackwardIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma
    if gamma === nothing
        gamma = 1 / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax)
    end

    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)

    state = FastForwardBackwardState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, z=z, g_z=g_z, res=x - z,
    )

    return state, state
end

function Base.iterate(iter::FastForwardBackwardIteration{R}, state::FastForwardBackwardState{R,Tx,TAx}) where {R,Tx,TAx}
    if iter.gamma === nothing || iter.adaptive == true
        state.gamma, state.g_z, _, _, _ = backtrack_stepsize!(
            state.gamma, iter.f, iter.A, iter.g,
            state.x, state.f_Ax, state.At_grad_f_Ax, state.y, state.z, state.g_z, state.res,
            minimum_gamma = iter.minimum_gamma,
        )
    end

    theta1 = (R(1) + sqrt(R(1) + 4 * state.theta^2)) / R(2)
    extr = (state.theta - R(1)) / theta1
    state.theta = theta1
    state.x .= state.z .+ extr .* (state.z .- state.z_prev)
    state.z_prev, state.z = state.z, state.z_prev
    
    # TODO: in the adaptive case we should be able to save some computation
    # by extrapolating Ax and (if f is quadratic) f_Ax, grad_f_Ax, At_grad_f_Ax.
    mul!(state.Ax, iter.A, state.x)
    state.f_Ax = gradient!(state.grad_f_Ax, iter.f, state.Ax)
    mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)
    state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
    state.res .= state.x .- state.z

    return state, state
end

# Solver

struct FastForwardBackward{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::FastForwardBackward)(x0; kwargs...)
    stop(state::FastForwardBackwardState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) =
        @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)
    iter = FastForwardBackwardIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.z, num_iters
end

FastForwardBackward(; maxit=10_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    FastForwardBackward(maxit, tol, verbose, freq, kwargs)
