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

@Base.kwdef struct ForwardBackwardIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,TA,Tg}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = false
    fast::Bool = false
end

Base.IteratorSize(::Type{<:ForwardBackwardIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct ForwardBackwardState{R,Tx,TAx}
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

f_model(state::ForwardBackwardState) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

function Base.iterate(iter::ForwardBackwardIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma

    if gamma === nothing
        # compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
        xeps = x .+ R(1)
        grad_f_Axeps, f_Axeps = gradient(iter.f, iter.A * xeps)
        L = norm(iter.A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
        gamma = R(1) / L
    end

    # compute initial forward-backward step
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)

    # compute initial fixed-point residual
    res = x - z

    state = ForwardBackwardState(; x, Ax, f_Ax, grad_f_Ax, At_grad_f_Ax, gamma, y, z, g_z, res)

    return state, state
end

function Base.iterate(iter::ForwardBackwardIteration{R}, state::ForwardBackwardState{R,Tx,TAx}) where {R,Tx,TAx}
    Az, f_Az, grad_f_Az, At_grad_f_Az = nothing, nothing, nothing, nothing
    a, b, c = nothing, nothing, nothing

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.gamma === nothing || iter.adaptive == true
        if state.gamma < 1e-7 # TODO: make this a parameter, or dependent on R?
            @warn "parameter `gamma` became too small ($(state.gamma)), stopping the iterations"
            return nothing
        end
        f_Az_upp = f_model(state)
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
    end

    if iter.fast == true
        theta1 = (R(1) + sqrt(R(1) + 4 * state.theta^2)) / R(2)
        extr = (state.theta - R(1)) / theta1
        state.theta = theta1
        state.x .= state.z .+ extr .* (state.z .- state.z_prev)
        state.z_prev, state.z = state.z, state.z_prev
    else
        state.x, state.z = state.z, state.x
    end

    # TODO: if iter.fast == true, in the adaptive case we should still be able
    # to save some computation by extrapolating Ax and (if f is quadratic)
    # f_Ax, grad_f_Ax, At_grad_f_Ax.
    if iter.fast == false && (iter.gamma === nothing || iter.adaptive == true)
        state.Ax = Az
        state.f_Ax = f_Az
        state.grad_f_Ax = grad_f_Az
    else
        mul!(state.Ax, iter.A, state.x)
        state.f_Ax = gradient!(state.grad_f_Ax, iter.f, state.Ax)
    end

    mul!(state.At_grad_f_Ax, adjoint(iter.A), state.grad_f_Ax)
    state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
    state.g_z = prox!(state.z, iter.g, state.y, state.gamma)

    state.res .= state.x .- state.z

    return state, state
end

# Solver

struct ForwardBackward{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::ForwardBackward)(x0; kwargs...)
    stop(state::ForwardBackwardState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) =
        @printf("%5d | %.3e | %.3e\n", it, state.gamma, norm(state.res, Inf) / state.gamma)
    iter = ForwardBackwardIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.z, num_iters
end

# Outer constructors

"""
    ForwardBackward([gamma, adaptive, fast, maxit, tol, verbose, freq])

Instantiate the Forward-Backward splitting algorithm (see [1, 2]) for solving
optimization problems of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).
If `solver = ForwardBackward(args...)`, then the above problem is solved with

    solver(x0, [f, A, g, L])

Optional keyword arguments:

* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `1/L` if not set (but `L` is).
* `adaptive::Bool` (default: `false`), if true, forces the method stepsize to be adaptively adjusted.
* `fast::Bool` (default: `false`), if true, uses Nesterov acceleration.
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.

If `gamma` is not specified at construction time, the following keyword
argument can be used to set the stepsize parameter:

* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).

References:

[1] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
Optimization" (2008).

[2] Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1,
pp. 183-202 (2009).
"""
ForwardBackward(; maxit=10_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    ForwardBackward(maxit, tol, verbose, freq, kwargs)
