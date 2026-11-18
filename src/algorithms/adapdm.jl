# Latafat, Themelis, Stella, Patrinos, "Adaptive proximal algorithms
# for convex optimization under local Lipschitz continuity of the gradient",
# Mathematical Programming, vol. 213, pp. 433-471 (2025).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalCore: Zero, convex_conjugate
using LinearAlgebra
using Printf

# TODO implement AdaPDMplus (Algorithm 3.2 of [1])

# TODO Default Lopnorm = opnorm(L)
# For a dense L this is a full SVD, which can be expensive at scale. It also fails or is 
# impractical for sparse or matrix-free operators. Document that users can pass Lopnorm 
# themselves, or use a power-iteration estimate as the default (with a safety factor, since 
# an underestimate breaks the stepsize bound).
"""
    AdaptivePrimalDualIteration(; <keyword-arguments>)

Iterator implementing the adaptive three-term primal-dual algorithm (AdaPDM, see [1]).

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x) + h(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth, and `L` is a linear mapping.

Points `x0` and `y0` are the initial primal and dual iterates, respectively.
If unspecified, functions `f`, `g`, and `h` default to the identically zero
function, and `L` defaults to the identity.

The iterator implements Algorithm 3.1 of [1]. A linesearch variant that does not require
the linear operator norm is given by Algorithm 3.2 of [1], not yet implemented.

See also: [`AdaptivePrimalDual`](@ref).

# Arguments
- `x0`: initial primal point.
- `y0`: initial dual point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: proximable objective term.
- `L=I`: linear operator (e.g. a matrix).
- `Lopnorm=opnorm(L)`: operator norm of `L`.
- `t=1`: positive algorithm parameter.
- `epsil=1e-6`: positive algorithm parameter.
- `nu=max(1.2,1+2*epsil)`: algorithm parameter such that `nu > 1 + epsil`.

# References
1. Latafat, Themelis, Stella, Patrinos, "Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient", Mathematical Programming, vol. 213, pp. 433-471 (2025).
"""
Base.@kwdef struct AdaptivePrimalDualIteration{R<:Real,Tx,Ty,Tf,Tg,Th,TL,Tnormop}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    L::TL = I
    x0::Tx
    y0::Ty
    Lopnorm::Tnormop = LinearAlgebra.opnorm(L) # TODO not provided? then linesearch variant AdaPDMplus
    t::R = real(eltype(x0))(1)
    epsil::R = real(eltype(x0))(1e-6)
    nu::R = max(real(eltype(x0))(1.2), 1+2*epsil)
end

Base.IteratorSize(::Type{<:AdaptivePrimalDualIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct AdaptivePrimalDualState{R,Tx,Ty}
    x::Tx               # primal iterate
    y::Ty               # dual iterate
    xbar::Tx
    ybar::Ty
    grad_f_x::Tx        # gradient of f at x
    grad_f_xold::Tx     # gradient of f at previous x
    L_x::Ty
    L_xold::Ty
    gamma::R
    gamma_old::R
    tmp_x::Tx
    tmp_y::Ty
    FPR_x::Tx           # fixed-point x-residual
    FPR_y::Ty           # fixed-point y-residual
end

function Base.iterate(iter::AdaptivePrimalDualIteration)
    iter.Lopnorm > 0 || throw(ArgumentError("`Lopnorm` must be positive, got $iter.Lopnorm"))
    iter.t > 0 || throw(ArgumentError("`t` must be positive, got $iter.t"))
    iter.epsil > 0 || throw(ArgumentError("`epsil` must be positive, got $iter.epsil"))
    iter.nu > 1 + iter.epsil || throw(ArgumentError("`nu` must satisfy nu > 1 + epsil, got nu = $iter.nu, epsil = $iter.epsil"))

    x = copy(iter.x0)
    y = copy(iter.y0)
    L_x = similar(y)
    mul!(L_x, iter.L, x)
    R = real(eltype(x))
    gamma = R(1/(2 * iter.nu * iter.t * iter.Lopnorm))

    state = AdaptivePrimalDualState(
        x=x,
        y=y,
        xbar=copy(x),
        ybar=copy(y),
        grad_f_x=similar(x),
        grad_f_xold=similar(x),
        L_x=L_x,
        L_xold=similar(y),
        gamma=gamma,
        gamma_old=copy(gamma),
        tmp_x=similar(x),
        tmp_y=similar(y),
        FPR_x=similar(x),
        FPR_y=similar(y),
    )

    _, state.grad_f_x = value_and_gradient(iter.f, state.x)

    # x-update
    mul!(state.tmp_x, iter.L', state.ybar)
    state.tmp_x .+= state.grad_f_x
    state.tmp_x .*= (- state.gamma)
    state.tmp_x .+= state.x
    prox!(state.xbar, iter.g, state.tmp_x, state.gamma)

    # residuals
    state.FPR_x .= state.xbar .- state.x
    state.FPR_y .= Inf

    return state, state
end

# stepsize rule as defined in Algorithm 3.1 of [1]
function get_stepsizes(iter::AdaptivePrimalDualIteration, gamma, x, grad_x, gamma_old, x_old, grad_x_old)
    return get_stepsizes(iter, gamma, gamma_old, x-x_old, grad_x-grad_x_old)
end

function get_stepsizes(iter::AdaptivePrimalDualIteration, gamma, gamma_old, delta_x, delta_grad_x)
    R = real(eltype(delta_x))
    t = iter.t
    nu = iter.nu
    epsil = iter.epsil
    Lopnorm = iter.Lopnorm

    # ℓ_k and c_k from (1.3) of [1]
    dx2 = norm(delta_x)^2
    l = real(dot(delta_grad_x, delta_x)) / dx2 |> nan_to_zero
    lc = norm(delta_grad_x)^2 / dx2 |> nan_to_zero

    # δ_k and ξ_k
    delta = gamma * (gamma*lc - l) |> nan_to_zero
    xi = (t*gamma*Lopnorm)^2

    # γ_k+1
    cc1 = gamma * sqrt(1 + gamma/gamma_old)
    cc2 = 1/(2*nu*t*Lopnorm)
    cc0 = 1-4*xi*(1+epsil)^2
    cc3 = gamma*sqrt(cc0/(2*(1+epsil)*(sqrt(delta^2 + xi*cc0) + delta)))
    gamma_new = R(min(cc1, cc2, cc3))
    return gamma_new, gamma
end

function Base.iterate(
    iter::AdaptivePrimalDualIteration,
    state::AdaptivePrimalDualState,
)
    state.grad_f_xold .= state.grad_f_x
    _, state.grad_f_x = value_and_gradient(iter.f, state.xbar)
    state.L_xold .= state.L_x
    mul!(state.L_x, iter.L, state.xbar)

    # stepsizes
    state.tmp_x .= state.xbar .- state.x # temporary
    state.FPR_x .= state.grad_f_x .- state.grad_f_xold # temporary
    state.gamma, state.gamma_old = get_stepsizes(iter, state.gamma, state.gamma_old, state.tmp_x, state.FPR_x)
    sigma = state.gamma * iter.t^2

    # x-residual
    # (tmp_x = xbar - x, FPR_x = grad_x - grad_xold at this point)
    state.FPR_x .-= state.tmp_x ./ state.gamma_old

    state.x .= state.xbar
    state.y .= state.ybar

    # y-update
    rho = state.gamma / state.gamma_old
    state.tmp_y .= (1+rho) .* state.L_x .- rho .* state.L_xold
    state.tmp_y .*= sigma
    state.tmp_y .+= state.y
    prox!(state.ybar, convex_conjugate(iter.h), state.tmp_y, sigma)

    # y-residual
    state.FPR_y .= (state.tmp_y .- state.ybar) ./ sigma .- state.L_x

    # x-update
    mul!(state.tmp_x, iter.L', state.ybar)
    state.tmp_x .+= state.grad_f_x
    state.tmp_x .*= (- state.gamma)
    state.tmp_x .+= state.x
    prox!(state.xbar, iter.g, state.tmp_x, state.gamma)

    return state, state
end

default_stopping_criterion(tol, ::AdaptivePrimalDualIteration, state::AdaptivePrimalDualState) =
    sqrt(norm(state.FPR_x)^2 + norm(state.FPR_y)^2) <= tol
default_solution(::AdaptivePrimalDualIteration, state::AdaptivePrimalDualState) = (state.x, state.ybar)
default_display(it, ::AdaptivePrimalDualIteration, state::AdaptivePrimalDualState) =
    @printf("%6d | %7.4e %7.4e\n", it, norm(state.FPR_x), norm(state.FPR_y))

"""
    AdaptivePrimalDual(; <keyword-arguments>)

Constructs the adaptive linesearch-free three-term primal-dual algorithm (AdaPDM, see [1]).

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x) + h(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth, and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{AdaptivePrimalDualIteration}`,
and can be called with the problem's arguments to trigger its solution.
The solution is returned as an (x, y) tuple.

See also: [`AdaptivePrimalDualIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol=1e-5`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AdaptivePrimalDualIteration` constructor upon call

# References
1. Latafat, Themelis, Stella, Patrinos, "Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient", Mathematical Programming, vol. 213, pp. 433-471 (2025).
"""
AdaptivePrimalDual(;
    maxit=10_000,
    tol=1e-5,
    stop=(iter, state) -> default_stopping_criterion(tol, iter, state),
    solution=default_solution,
    verbose=false,
    freq=100,
    display=default_display,
    kwargs...,
) = IterativeAlgorithm(
    AdaptivePrimalDualIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    display,
    kwargs...,
)


# Aliases

const AdaPDMState = AdaptivePrimalDualState
const AdaPDMIteration = AdaptivePrimalDualIteration
const AdaPDM = AdaptivePrimalDual