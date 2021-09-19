# Lions, Mercier, “Splitting algorithms for the sum of two nonlinear
# operators,” SIAM Journal on Numerical Analysis, vol. 16, pp. 964–979 (1979).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    ForwardBackwardIteration(; <keyword-arguments>)

Instantiate the forward-backward splitting algorithm (see [1, 2]) for solving
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
- [1] Lions, Mercier, “Splitting algorithms for the sum of two nonlinear
operators,” SIAM Journal on Numerical Analysis, vol. 16, pp. 964–979 (1979).
"""

Base.@kwdef struct ForwardBackwardIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,TA,Tg}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (1 / Lf)
    adaptive::Bool = false
    minimum_gamma::R = real(eltype(x0))(1e-7)
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
end

f_model(state::ForwardBackwardState) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, 1 / state.gamma)

function Base.iterate(iter::ForwardBackwardIteration{R}) where {R}
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

    state = ForwardBackwardState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, z=z, g_z=g_z, res=x - z,
    )

    return state, state
end

function Base.iterate(iter::ForwardBackwardIteration{R}, state::ForwardBackwardState{R,Tx,TAx}) where {R,Tx,TAx}
    if iter.gamma === nothing || iter.adaptive == true
        state.gamma, state.g_z, Az, f_Az, grad_f_Az = backtrack_stepsize!(
            state.gamma, iter.f, iter.A, iter.g,
            state.x, state.f_Ax, state.At_grad_f_Ax, state.y, state.z, state.g_z, state.res,
            minimum_gamma = iter.minimum_gamma,
        )
        state.x, state.z = state.z, state.x
        state.Ax = Az
        state.f_Ax = f_Az
        state.grad_f_Ax = grad_f_Az
    else
        state.x, state.z = state.z, state.x
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

ForwardBackward(; maxit=10_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    ForwardBackward(maxit, tol, verbose, freq, kwargs)
