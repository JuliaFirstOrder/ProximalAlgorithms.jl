using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

struct FBS_iterable{R <: Real, C <: Union{R, Complex{R}}, Tx <: AbstractArray{C}, Tf, TA, Tg}
    f::Tf             # smooth term
    A::TA             # matrix/linear operator
    g::Tg             # (possibly) nonsmooth, proximable term
    x0::Tx            # initial point
    gamma::Maybe{R}   # stepsize parameter of forward and backward steps
    adaptive::Bool    # enforce adaptive stepsize even if L is provided
    fast::Bool
end

mutable struct FBS_state{R <: Real, Tx, TAx}
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
    theta::R
    z_prev::Tx
end

f_model(state::FBS_state) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

function Base.iterate(iter::FBS_iterable{R}) where R
    x = iter.x0
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma

    if gamma === nothing
        # compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
        xeps = x .+ R(1)
        grad_f_Axeps, f_Axeps = gradient(iter.f, iter.A*xeps)
        L = norm(iter.A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
        gamma = R(1)/L
    end

    # compute initial forward-backward step
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(iter.g, y, gamma)

    # compute initial fixed-point residual
    res = x - z

    state = FBS_state(x, Ax, f_Ax, grad_f_Ax, At_grad_f_Ax, gamma, y, z, g_z, res, R(1), copy(x))

    return state, state
end

function Base.iterate(iter::FBS_iterable{R}, state::FBS_state{R, Tx, TAx}) where {R, Tx, TAx}
    Az, f_Az, grad_f_Az, At_grad_f_Az = nothing, nothing, nothing, nothing
    a, b, c = nothing, nothing, nothing

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.gamma === nothing || iter.adaptive == true
        if state.gamma < 1e-7 # TODO: make this a parameter, or dependent on R?
            @warn "parameter `gamma` became too small ($(state.gamma)), stopping the iterations"
            return nothing
        end
        f_Az_upp = f_model(state)
        Az = iter.A*state.z
        grad_f_Az, f_Az = gradient(iter.f, Az)
        tol = 10*eps(R)*(1 + abs(f_Az))
        if f_Az <= f_Az_upp + tol break end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_z = prox!(state.z, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.z
    end

    if iter.fast == true
        theta1 = (R(1)+sqrt(R(1)+4*state.theta^2))/R(2)
        extr = (state.theta - R(1))/theta1
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

"""
    forwardbackward(x0; f, A, g, [...])

Minimizes f(A*x) + g(x) with respect to x, starting from x0, using the
forward-backward splitting algorithm (also known as proximal gradient method).
If unspecified, f and g default to the identically zero function, while A
defaults to the identity.

Other optional keyword arguments:

* `L::Real` (default: `nothing`), the Lipschitz constant of the gradient of x ↦ f(Ax).
* `gamma::Real` (default: `nothing`), the stepsize to use; defaults to `1/L` if not set (but `L` is).
* `adaptive::Bool` (default: `false`), if true, forces the method stepsize to be adaptively adjusted.
* `fast::Bool` (default: `false`), if true, uses Nesterov acceleration.
* `maxit::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10`), frequency of verbosity.

References:

[1] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave
Optimization" (2008).

[2] Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm
for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1,
pp. 183-202 (2009).
"""
function forwardbackward(x0;
    f=Zero(), A=I, g=Zero(),
    L=nothing, gamma=nothing,
    adaptive=false, fast=false,
    maxit=10_000, tol=1e-8,
    verbose=false, freq=100)

    R = real(eltype(x0))

    stop(state::FBS_state) = norm(state.res, Inf)/state.gamma <= R(tol)
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e\n",
        it, state.gamma, norm(state.res, Inf)/state.gamma
    )

    if gamma === nothing && L !== nothing
        gamma = R(1)/R(L)
    elseif gamma !== nothing
        gamma = R(gamma)
    end

    iter = FBS_iterable(f, A, g, x0, gamma, adaptive, fast)
    iter = take(halt(iter, stop), maxit)
    iter = enumerate(iter)
    if verbose iter = tee(sample(iter, freq), disp) end

    num_iters, state_final = loop(iter)

    return state_final.z, num_iters
end
