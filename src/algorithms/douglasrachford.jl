################################################################################
# Douglas-Rachford splitting iterable
#
# [1] Eckstein, Bertsekas "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators*",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
#

using Base.Iterators
using ProximalAlgorithms: LBFGS
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

struct DRS_iterable{R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}, Tf, Tg}
    f::Tf
    g::Tg
    x::T
    gamma::R
end

mutable struct DRS_state{T}
    x::T
    y::T
    r::T
    z::T
    res::T
end

DRS_state(iter::DRS_iterable) = DRS_state(copy(iter.x), zero(iter.x), zero(iter.x), zero(iter.x), zero(iter.x))

function Base.iterate(iter::DRS_iterable, state::DRS_state=DRS_state(iter))
    prox!(state.y, iter.f, state.x, iter.gamma)
    state.r .= 2 .*state.y .- state.x
    prox!(state.z, iter.g, state.r, iter.gamma)
    state.res .= state.y .- state.z
    state.x .-= state.res
    return state, state
end

struct DRS{R}
    gamma::R
    maxit
    tol
    verbose
    freq

    function DRS{R}(; gamma::R, maxit::Int=1000, tol::R=R(1e-8),
        verbose::Bool=false, freq::Int=100
    ) where R
        @assert gamma > 0
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(gamma, maxit, tol, verbose, freq)
    end
end

function (solver::DRS{R})(
    x0::AbstractArray{C}; f=Zero(), g=Zero()
) where {R, C <: Union{R, Complex{R}}}
    stop(state::DRS_state) = norm(state.res, Inf) <= solver.tol
    disp((it, state)) = @printf("%5d | %.3e\n", it, norm(state.res, Inf))

    iter = DRS_iterable(f, g, x0, solver.gamma)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose iter = tee(sample(iter, solver.freq), disp) end

    num_iters, state_final = loop(iter)

    return state_final.y, state_final.z, num_iters
end

# """
#     douglasrachford(x0; f, g, gamma, [...])
#
# Minimizes `f(x) + g(x)` with respect to `x`, using the Douglas-Rachfor splitting
# algorithm starting from `x0`, with stepsize `gamma`.
# If unspecified, `f` and `g` default to the identically zero function,
# while `gamma` defaults to one.
#
# Other optional keyword arguments:
#
# * `maxit::Integer` (default: `1000`), maximum number of iterations to perform.
# * `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
# * `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
# * `freq::Integer` (default: `100`), frequency of verbosity.
#
# References:
#
# [1] Eckstein, Bertsekas, "On the Douglas-Rachford Splitting Method and the
# Proximal Point Algorithm for Maximal Monotone Operators",
# Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
# """
# function douglasrachford(x0;
#     f=Zero(), g=Zero(),
#     gamma=1.0,
#     maxit=1000, tol=1e-8,
#     verbose=false, freq=100)
#
#     R = real(eltype(x0))
#
#     stop(state::DRS_state) = norm(state.res, Inf) <= R(tol)
#     disp((it, state)) = @printf("%5d | %.3e\n", it, norm(state.res, Inf))
#
#     iter = DRS_iterable(f, g, x0, R(gamma))
#     iter = take(halt(iter, stop), maxit)
#     iter = enumerate(iter)
#     if verbose iter = tee(sample(iter, freq), disp) end
#
#     num_iters, state_final = loop(iter)
#
#     return state_final.y, state_final.z, num_iters
# end
