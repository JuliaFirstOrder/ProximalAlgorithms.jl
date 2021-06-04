# Latafat, Patrinos, "Asymmetric forward–backward–adjoint splitting for
# solving monotone inclusions involving three operators", Computational
# Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017).
#
# Latafat, Patrinos, "Primal-dual proximal algorithms for structured convex
# optimization: a unifying framework", In Large-Scale and Distributed 
# Optimization, Giselsson and Rantzer, Eds. Springer International Publishing,
# pp. 97–120 (2018).
#
# Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
# with Applications to Imaging", Journal of Mathematical Imaging and Vision,
# vol. 40, no. 1, pp. 120-145 (2011).
#
# Condat, "A primal–dual splitting method for convex optimization
# involving Lipschitzian, proximable and linear composite terms",
# Journal of Optimization Theory and Applications, vol. 158, no. 2,
# pp 460-479 (2013).
# 
# Vũ, "A splitting algorithm for dual monotone inclusions involving
# cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3,
# pp. 667-681 (2013).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

Base.@kwdef struct AFBA_iterable{R,Tx,Ty,Tf,Tg,Th,Tl,TL}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    l::Tl = IndZero()
    L::TL = if isa(h, Zero) L = 0 * I else I end
    x0::Tx
    y0::Ty
    beta_f::R = real(eltype(x0))(0)
    beta_l::R = real(eltype(x0))(0)
    theta = 1
    mu = 1
    lambda::R = real(eltype(x0))(1)
    gamma::Tuple{R, R} = begin
        if lambda != 1
            @warn "default stepsizes are not supported with this choice of lamdba, reverted to default lambda"
            lambda = real(eltype(x0))(1)
        end
        AFBA_default_stepsizes(L, h, theta, mu, beta_f, beta_l)
    end
end

Base.IteratorSize(::Type{<:AFBA_iterable}) = Base.IsInfinite()

struct AFBA_state{Tx,Ty}
    x::Tx
    y::Ty
    xbar::Tx
    ybar::Ty
    gradf::Tx
    gradl::Ty
    FPR_x::Tx
    FPR_y::Ty
    temp_x::Tx
    temp_y::Ty
end

AFBA_state(iter::AFBA_iterable) = AFBA_state(
    copy(iter.x0),
    copy(iter.y0),
    zero(iter.x0),
    zero(iter.y0),
    zero(iter.x0),
    zero(iter.y0),
    zero(iter.x0),
    zero(iter.y0),
    zero(iter.x0),
    zero(iter.y0),
)

function Base.iterate(iter::AFBA_iterable, state::AFBA_state = AFBA_state(iter))
    # perform xbar-update step
    gradient!(state.gradf, iter.f, state.x)
    mul!(state.temp_x, iter.L', state.y)
    state.temp_x .+= state.gradf
    state.temp_x .*= -iter.gamma[1]
    state.temp_x .+= state.x
    prox!(state.xbar, iter.g, state.temp_x, iter.gamma[1])

    # perform ybar-update step
    gradient!(state.gradl, Conjugate(iter.l), state.y)
    state.temp_x .= iter.theta .* state.xbar .+ (1 - iter.theta) .* state.x
    mul!(state.temp_y, iter.L, state.temp_x)
    state.temp_y .-= state.gradl
    state.temp_y .*= iter.gamma[2]
    state.temp_y .+= state.y
    prox!(state.ybar, Conjugate(iter.h), state.temp_y, iter.gamma[2])

    # the residues
    state.FPR_x .= state.xbar .- state.x
    state.FPR_y .= state.ybar .- state.y

    # perform x-update step
    state.temp_y .= (iter.mu * (2 - iter.theta) * iter.gamma[1]) .* state.FPR_y
    mul!(state.temp_x, iter.L', state.temp_y)
    state.x .+= iter.lambda .* (state.FPR_x .- state.temp_x)

    # perform y-update step
    state.temp_x .= ((1 - iter.mu) * (2 - iter.theta) * iter.gamma[2]) .* state.FPR_x
    mul!(state.temp_y, iter.L, state.temp_x)
    state.y .+= iter.lambda .* (state.FPR_y .+ state.temp_y)

    return state, state
end

# Solver

struct AFBA{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::AFBA)(x0, y0; kwargs...)
    stop(state::AFBA_state) = norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf) <= solver.tol
    disp((it, state)) =
        @printf("%6d | %7.4e\n", it, norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf))
    iter = AFBA_iterable(; x0=x0, y0=y0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.xbar, state_final.ybar, num_iters
end

# Outer constructors

"""
    AFBA([gamma1, gamma2, theta, mu, lambda, maxit, tol, verbose, freq])

Instantiate the asymmetric forward-backward-adjoing algorithm (AFBA, see [1])
for solving convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly
convex. Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.
If `solver = AFBA(args...)`, then the above problem is solved with

    solver(x0, y0; [f, g, h, l, L, beta_f, beta_l])

Points `x0` and `y0` are the initial primal and dual iterates, respectively.
If unspecified, functions `f`, `g`, and `h` default to the identically zero
function, `l` defaults to the indicator of the set `{0}`, and `L` defaults to
the identity. Important keyword arguments, in case `f` and `l` are set, are:

* `beta_f`: Lipschitz constant of gradient of `f` (default: zero)
* `beta_l`: Lipschitz constant of gradient of the conjugate of `l` (default: zero)

These are used to determine the algorithm default stepsizes, `gamma1` and `gamma2`,
in case they are not directly specified.

Optional keyword arguments are:

* `gamma1`: stepsize corresponding to the primal updates (default: see [1] for each case)
* `gamma2`: stepsize corresponding to the dual updates (default: see [1] for each case)
* `theta`: nonnegative algorithm parameter (default: `1.0`)
* `mu`: algorithm parameter in the range [0,1] (default: `1.0`)
* `tol`: primal-dual termination tolerance (default: `1e-5`)
* `maxit`: maximum number of iterations (default: `10000`)
* `verbose`, verbosity level (default: `1`)
* `verbose_freq`, verbosity frequency for `verbose = 1` (default: `100`)

The iterator implements Algorithm 3 of [1] with constant stepsize (α_n=λ)
for several prominant special cases:
1) θ = 2          ==>   Corresponds to the Vu-Condat Algorithm [2,3].
2) θ = 1, μ=1
3) θ = 0, μ=1
4) θ ∈ [0,∞), μ=0

See [2, Section 5.2] and [1, Figure 1] for stepsize conditions, special cases,
and relation to other algorithms.

References:

[1] Latafat, Patrinos, "Asymmetric forward–backward–adjoint splitting for
solving monotone inclusions involving three operators", Computational
Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017).

[2] Latafat, Patrinos, "Primal-dual proximal algorithms for structured convex
optimization : a unifying framework", In Large-Scale and Distributed 
Optimization, Giselsson and Rantzer, Eds. Springer International Publishing,
pp. 97–120 ( 2018).

[3] Condat, "A primal–dual splitting method for convex optimization
involving Lipschitzian, proximable and linear composite terms",
Journal of Optimization Theory and Applications, vol. 158, no. 2,
pp 460-479 (2013).

[4] Vũ, "A splitting algorithm for dual monotone inclusions involving
cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3,
pp. 667-681 (2013).
"""
AFBA(; maxit=10_000, tol=1e-5, verbose=false, freq=100, kwargs...) = 
    AFBA(maxit, tol, verbose, freq, kwargs)


"""
    VuCondat([gamma1, gamma2, theta, mu, lambda, maxit, tol, verbose, freq])

Instantiate the Vû-Condat splitting algorithm (see [2, 3])
for solving convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly
convex. Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.
If `solver = VuCondat(args...)`, then the above problem is solved with

    solver(x0, y0; [f, g, h, l, L, beta_f, beta_l])

See documentation of `AFBA` for the list of keyword arguments.

References:

[1] Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
with Applications to Imaging", Journal of Mathematical Imaging and Vision,
vol. 40, no. 1, pp. 120-145 (2011).

[2] Condat, "A primal–dual splitting method for convex optimization
involving Lipschitzian, proximable and linear composite terms",
Journal of Optimization Theory and Applications, vol. 158, no. 2,
pp 460-479 (2013).

[3] Vũ, "A splitting algorithm for dual monotone inclusions involving
cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3,
pp. 667-681 (2013).
"""
VuCondat(; maxit=10_000, tol=1e-5, verbose=false, freq=100, kwargs...) = 
    AFBA(maxit=maxit, tol=tol, verbose=verbose, freq=freq, kwargs..., theta=2)

function AFBA_default_stepsizes(L, h, theta, mu, beta_f::R, beta_l::R) where {R<:Real}
    # lambda = 1
    if isa(h, Zero)
        gamma1 = R(1.99) / beta_f
        gamma2 = R(1) # does not matter
    else
        par = R(5) # scaling parameter for comparing Lipschitz constants and \|L\|
        par2 = R(100)   # scaling parameter for α
        alpha = R(1)
        nmL = R(opnorm(L))
        if theta == 2 # default stepsize for Vu-Condat
            if nmL > par * max(beta_l, beta_f)
                alpha = R(1)
            elseif beta_f > par * beta_l
                alpha = par2 * nmL / beta_f
            elseif beta_l > par * beta_f
                alpha = beta_l / (par2 * nmL)
            end
            gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
        elseif theta == 1 && mu == 1 # SPCA
            if nmL > par2 * beta_l # for the case beta_f = 0
                alpha = R(1)
            elseif beta_l > par * beta_f
                alpha = beta_l / (par2 * nmL)
            end
            gamma1 = beta_f > 0 ? R(1.99) / beta_f : R(1) / (nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + gamma1 * nmL^2)
        elseif theta == 0 && mu == 1 # PPCA
            temp = R(3)
            if beta_f == 0
                nmL *= sqrt(temp)
                if nmL > par * beta_l
                    alpha = R(1)
                else
                    alpha = beta_l / (par2 * nmL)
                end
                gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
                gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
            else
                if nmL > par * max(beta_l, beta_f)
                    alpha = R(1)
                elseif beta_f > par * beta_l
                    alpha = par2 * nmL / beta_f
                elseif beta_l > par * beta_f
                    alpha = beta_l / (par2 * nmL)
                end
                xi = 1 + 2 * nmL / (nmL + alpha * beta_f / 2)
                gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
                gamma2 = R(0.99) / (beta_l / 2 + xi * nmL * alpha)
            end
        elseif mu == 0 # SDCA & PDCA
            temp = theta^2 - 3 * theta + 3
            if beta_l == R(0)
                nmL *= sqrt(temp)
                if nmL > par * beta_f
                    alpha = R(1)
                else
                    alpha = par2 * nmL / beta_f
                end
                gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
                gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
            else
                if nmL > par * max(beta_l, beta_f)
                    alpha = R(1)
                elseif beta_f > par * beta_l
                    alpha = par2 * nmL / beta_f
                elseif beta_l > par * beta_f
                    alpha = beta_l / (par2 * nmL)
                end
                eta = 1 + (temp - 1) * alpha * nmL / (alpha * nmL + beta_l / 2)
                gamma1 = R(1) / (beta_f / 2 + eta * nmL / alpha)
                gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
            end
        elseif theta == 0 && mu == 0.5 # PPDCA
            if beta_l == 0 || beta_f == 0
                if nmL > par * max(beta_l, beta_f)
                    alpha = R(1)
                elseif beta_f > par * beta_l
                    alpha = par2 * nmL / beta_f
                elseif beta_l > par * beta_f
                    alpha = beta_l / (par2 * nmL)
                end
            else
                alpha = sqrt(beta_l / beta_f) / 2
            end
            gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
        else
            error("this choice of theta and mu is not supported!")
        end
    end

    return gamma1, gamma2
end