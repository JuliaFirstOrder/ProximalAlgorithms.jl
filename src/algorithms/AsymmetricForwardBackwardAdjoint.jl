################################################################################
# Primal-dual algorithms based on Asymmetric Forward-Backward-Adjoint
#
# [1] Latafat, Patrinos. "Asymmetric forward–backward–adjoint splitting for solving monotone inclusions involving three operators"  Computational Optimization and Applications, pages 1–37, 2017.
# [2] Condat. "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms" Journal of Optimization Theory and Applications 158.2 (2013): 460-479.
# [3] Vũ. "A splitting algorithm for dual monotone inclusions involving cocoercive operators"" Advances in Computational Mathematics, 38(3), pp.667-681.

struct AFBAIterator{I <: Integer, R <: Real, T1 <: AbstractArray, T2 <: AbstractArray} <: ProximalAlgorithm{I, Tuple{T1, T2}}
    x::T1
    y::T2
    g
    hconj
    f
    lconj
    L
    theta::R
    mu::R
    lam::R
    betaQ::R # Lipschitz constant of gradient of f
    betaR::R # Lipschitz constant of gradient of l conjugate
    gamma1::R
    gamma2::R
    maxit::I
    tol::R
    verbose::I
    verbose_freq::I
    xbar::T1
    ybar::T2
    gradf::T1
    gradl::T2
    FPR_x::T1
    FPR_y::T2
    temp_x::T1
    temp_y::T2
end

################################################################################
# Constructor(s)

function AFBAIterator(x0::T1, y0::T2; g=IndFree(), h=IndFree(), f=IndFree(), l=IndZero(), L=Identity(size(x0)), theta=1, mu=1, lam=1, betaQ=0, betaR=0, gamma1=-1, gamma2=-1, maxit::I=10000, tol::R=1e-5, verbose=1, verbose_freq = 100) where {I, R, T1, T2}
    x = copy(x0)
    xbar = copy(x0)
    y = copy(y0)
    ybar = copy(y0)
    gradf = copy(x0)
    gradl = copy(y0)
    FPR_x = copy(x0)
    FPR_x .= Inf
    FPR_y = copy(y0)
    FPR_y .= Inf
    temp_x = copy(x0)
    temp_y = copy(y0)
    hconj = Conjugate(h)
    lconj = Conjugate(l)

    # default stepsizes
    par = 4 #  scale parameter for comparing Lipschitz constant and opnorm(L)
    nmL = opnorm(L)
    alpha = 1
    if isa(h, ProximalOperators.IndFree) && (gamma1<0 || gamma2<0)
        # mu=0 is the only case where stepsizes matter
        alpha = 1000/(betaQ+1e-5) # the speed is determined by gamma1 since bary=0
        temp = theta^2-3*theta+3
        gamma1 = 0.99/(betaQ/2+ temp*nmL/alpha) # in this case R=0
        gamma2 = 1/(nmL*alpha)
    elseif gamma1<0 || gamma2<0
        if theta==2 # default stepsize for Vu-Condat
            if betaQ > par*nmL && betaR > par*nmL
                alpha = 1
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ
            elseif betaR > par*nmL
                alpha = betaR/(2*nmL)
            end
            gamma1 = 1/(betaQ/2+nmL/alpha)
            gamma2 = 0.99/(betaR/2+nmL*alpha)
        elseif theta==1 && mu==1 # default stepsize for theta=1, mu=1
            if betaQ > par*nmL && betaR > par*nmL
                alpha = 1
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ
            elseif  betaR > par*nmL
                alpha = betaR/(2*nmL)
            end
            gamma1 = 1/(betaQ/2+nmL/alpha)
            gamma2 = 0.99/(betaR/2+gamma1*nmL^2)
        elseif theta==0 && mu==1 # default stepsize for theta=0, mu=1
            temp=3
            if betaQ > par*nmL && betaR > temp*par*nmL # denominator for Sigma involves 3α opnorm(L) in this case
                alpha = 1
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ
            elseif betaR > temp*par*nmL # denominator for Sigma involves 3α opnorm(L) in this case
                alpha = betaR/(temp*2*nmL)
            end
            gamma1 = 1/(betaQ/2+nmL/alpha)
            gamma2 = 0.99/(betaR/2+2*gamma1*nmL^2+ alpha*nmL)
        elseif mu==0 # default stepsize for  mu=0
            temp = theta^2-3*theta+3
            if betaQ > temp*par*nmL && betaR > par*nmL
                alpha = 1
            elseif betaR > par*nmL
                alpha = betaR/(2*nmL)
            elseif betaQ > temp*par*nmL # denominator for Sigma involves 3α opnorm(L) in this case
                alpha = 2*nmL*temp/betaQ
            end
            gamma2 = 1/(betaR/2+ alpha*nmL)
            gamma1 = 0.99/(betaQ/2+ (temp-1)*gamma2*nmL^2+ nmL/alpha)
        else
            error("this choice of theta and mu is not supported!")
        end
    end
    return AFBAIterator{I, R, T1, T2}(x0, y0, g, hconj, f, lconj, L, theta, mu, lam, betaQ, betaR, gamma1, gamma2, maxit, tol, verbose, verbose_freq, xbar, ybar, gradf, gradl, FPR_x,FPR_y, temp_x, temp_y)
end

################################################################################
# Utility methods
maxit(sol::AFBAIterator) = sol.maxit

converged(sol::AFBAIterator, it) = it > 0 && norm(sol.FPR_x)+norm(sol.FPR_y) <= sol.tol

verbose(sol::AFBAIterator) = sol.verbose > 0
verbose(sol::AFBAIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::AFBAIterator)
    @printf("%6s | %22s | %10s |\n ", "it", "gamma", "fpr")
    @printf("------|------------------------|------------|\n")
end

function display(sol::AFBAIterator, it)
    @printf("%6d | %7.4e, %7.4e | %7.4e |\n", it, sol.gamma1, sol.gamma2, norm(sol.FPR_x)+norm(sol.FPR_y))
end

function Base.show(io::IO, sol::AFBAIterator)
    println(io, "Asymmetric Forward-Backward-Adjoint Splitting" )
    if sol.theta==2
        println(io, "theta               : $(sol.theta)" )
    else
        println(io, "theta, mu           : $(sol.theta), $(sol.mu)" )
    end
    println(io, "fpr                 : $(norm(sol.FPR_x)+norm(sol.FPR_y))")
    print(  io, "gamma1, gamma2      : $(sol.gamma1), $(sol.gamma2)")
end

################################################################################
# Initialization

function initialize!(sol::AFBAIterator)
    return sol.x, sol.y
end

################################################################################
# Iteration

function iterate!(sol::AFBAIterator{I, R, T1, T2}, it::I) where {I, R, T1, T2}
    # perform xbar-update step
    gradient!(sol.gradf, sol.f, sol.x)
    mul!(sol.temp_x, sol.L', sol.y)
    sol.temp_x .+= sol.gradf
    sol.temp_x .*= -sol.gamma1
    sol.temp_x .+= sol.x
    prox!(sol.xbar, sol.g, sol.temp_x, sol.gamma1)
    # perform ybar-update step
    gradient!(sol.gradl, sol.lconj, sol.y)
    sol.temp_x .=  (sol.theta * sol.xbar) .+ ((1-sol.theta) * sol.x)
    mul!(sol.temp_y, sol.L, sol.temp_x)
    sol.temp_y .-= sol.gradl
    sol.temp_y .*= sol.gamma2
    sol.temp_y .+= sol.y
    prox!(sol.ybar, sol.hconj, sol.temp_y, sol.gamma2)
    # the residues
    sol.FPR_x .= sol.xbar .- sol.x
    sol.FPR_y .= sol.ybar .- sol.y
    # perform x-update step
    sol.temp_y .= sol.mu*(2-sol.theta)*sol.gamma1*sol.FPR_y
    mul!(sol.temp_x, sol.L', sol.temp_y)
    sol.x .+= sol.lam *(sol.FPR_x .- sol.temp_x)
    # perform y-update step
    sol.temp_x .= (1-sol.mu)*(2-sol.theta)*sol.gamma2*sol.FPR_x
    mul!(sol.temp_y, sol.L, sol.temp_x)
    sol.y .+= sol.lam *(sol.FPR_y .+ sol.temp_y)
    return sol.x, sol.y
end

################################################################################
# Solver interface(s)

"""
**Asymmetric forward-backward-adjoint algorithm**

    AFBA(x0, y0; kwargs)

Solves convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly convex.
Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.
Points `x0` and `y0` are the initial primal and dual iterates, respectively.

Keyword arguments are as follows:
* `f`: smooth, convex function (default: zero)
* `g`: convex function (possibly nonsmooth, default: zero)
* `h`: convex function (possibly nonsmooth, default: zero)
* `l`: strongly convex function (possibly nonsmooth, default: indicator of {0})
* `L`: linear operator (default: identity)
* `betaQ`: Lipschitz constant of gradient of f (default: zero)
* `betaR`: Lipschitz constant of gradient of l conjugate (default: zero)
* `gamma1`: stepsize corresponding to the primal updates (default: see [1] for each case)
* `gamma2`: stepsize corresponding to the dual updates (default: see [1] for each case)
* `theta`: nonnegative algorithm parameter (default: '1')
* `mu`: algorithm parameter in the range [0,1] (default: '1')
* `tol`: primal-dual termination tolerance (default: `1e-5`)
* `maxit`: maximum number of iterations (default: `10000`)
* `verbose`, verbosity level (default: `1`)
* `verbose_freq`, verbosity frequency for `verbose = 1` (default: `100`)

The iterator implements Algorithm 3 of [1] with constant stepsize (α_n=λ) for several prominant special cases:
1) θ = 2          ==>   Corresponds to the Vu-Condat Algorithm [2,3].
2) θ = 1, μ=1
3) θ = 0, μ=1
4) θ ∈ [0,∞), μ=0

See [1, Figure 1] for other special cases and relation to other algorithms.

[1] Latafat, Patrinos. "Asymmetric forward–backward–adjoint splitting for solving monotone inclusions involving three operators"  Computational Optimization and Applications, pages 1–37, 2017.
[2] Condat. "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms" Journal of Optimization Theory and Applications 158.2 (2013): 460-479.
[3] Vũ. "A splitting algorithm for dual monotone inclusions involving cocoercive operators"" Advances in Computational Mathematics, 38(3), pp.667-681.
"""
function AFBA(x0, y0; kwargs...)
    # Create iterable
    sol = AFBAIterator(x0, y0; kwargs...)
    # Run iterations
    (it, (primal, dual)) = run!(sol)
    return (it, primal, dual, sol)
end

"""
**Vũ-Condat primal-dual algorithm**

    VuCondat(x0, y0; kwargs)

Solves convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x).

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly convex.

Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.
Points `x0` and `y0` are the initial primal and dual iterates, respectively.

See documentation of `AFBA` for the list of keyword arguments.
"""
function VuCondat(x0, y0; kwargs...)
    # Create iterable
    sol = AFBAIterator(x0, y0; kwargs..., theta=2)
    # Run iterations
    (it, (primal, dual)) = run!(sol)
    return (it, primal, dual, sol)
end

"""
**Chambolle-Pock primal-dual algorithm**

    ChambollePock(x0, y0; kwargs)

Solves convex optimization problems of the form

    minimize g(x) + (h □ l)(L x).

where `g` and `h` are possibly nonsmooth and `l` is strongly convex.
Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.
Points `x0` and `y0` are the initial primal and dual iterates, respectively.

See documentation of `AFBA` for the list of keyword arguments.
"""
function ChambollePock(x0, y0; kwargs...)
    # Create iterable
    sol = AFBAIterator(x0, y0; kwargs..., f=IndFree(), theta=2)
    # Run iterations
    (it, (primal, dual)) = run!(sol)
    return (it, primal, dual, sol)
end
