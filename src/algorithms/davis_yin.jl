# Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization
# Applications", Set-Valued and Variational Analysis, vol. 25, no. 4,
# pp. 829–858 (2017).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    DavisYinIteration(; <keyword-arguments>)

Instantiate the Davis-Yin splitting algorithm (see [1]) for solving
convex optimization problems of the form

    minimize f(x) + g(x) + h(x),

where `h` is smooth.

# Arguments
- `x0`: initial point.
- `f=Zero()`: proximable objective term.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: smooth objective term.
- `Lh=nothing`: Lipschitz constant of the gradient of h.
- `gamma=nothing`: stepsize to use, defaults to `1/Lh` if not set (but `Lh` is).

# References
- [1] Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization
Applications", Set-Valued and Variational Analysis, vol. 25, no. 4,
pp. 829–858 (2017).
"""

Base.@kwdef struct DavisYinIteration{R,C<:Union{R,Complex{R}},T<:AbstractArray{C},Tf,Tg,Th}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    x0::T
    lambda::R = real(eltype(x0))(1)
    Lh::Maybe{R} = nothing
    gamma::Maybe{R} = Lh !== nothing ? (1 / Lh) : error("You must specify either Lh or gamma")
end

Base.IteratorSize(::Type{<:DavisYinIteration}) = Base.IsInfinite()

struct DavisYinState{T}
    z::T
    xg::T
    grad_h_xg::T
    z_half::T
    xf::T
    res::T
end

function Base.iterate(iter::DavisYinIteration)
    z = copy(iter.x0)
    xg, = prox(iter.g, z, iter.gamma)
    grad_h_xg, = gradient(iter.h, xg)
    z_half = 2 .* xg .- z .- iter.gamma .* grad_h_xg
    xf, = prox(iter.f, z_half, iter.gamma)
    res = xf - xg
    z .+= iter.lambda .* res
    state = DavisYinState(z, xg, grad_h_xg, z_half, xf, res)
    return state, state
end

function Base.iterate(iter::DavisYinIteration, state::DavisYinState)
    prox!(state.xg, iter.g, state.z, iter.gamma)
    gradient!(state.grad_h_xg, iter.h, state.xg)
    state.z_half .= 2 .* state.xg .- state.z .- iter.gamma .* state.grad_h_xg
    prox!(state.xf, iter.f, state.z_half, iter.gamma)
    state.res .= state.xf .- state.xg
    state.z .+= iter.lambda .* state.res
    return state, state
end

# Solver

struct DavisYin{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::DavisYin)(x0; kwargs...)
    stop(state::DavisYinState) = norm(state.res, Inf) <= solver.tol
    disp((it, state)) = @printf("%5d | %.3e\n", it, norm(state.res, Inf))
    iter = DavisYinIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.xf, state_final.xg, num_iters
end

DavisYin(; maxit=10_000, tol=1e-8, verbose=false, freq=100, kwargs...) = 
    DavisYin(maxit, tol, verbose, freq, kwargs)
