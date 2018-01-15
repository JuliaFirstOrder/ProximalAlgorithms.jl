################################################################################
# Primal-dual algorithms based on Asymmetric Forward-Backward-Adjoint that solve the following convex problem:
#
#								minimize_x f(x) + g(x) + (h □ l)(Lx), 
#
# where f is smooth, g and h are possibly nonsmooth and l is strongly convex. The infimal convolution is denoted by '□', and L is a linear mapping.   
#
# Usage:	 it, x, sol = ProximalAlgorithms.AFBA(x0, y0; g, h, f, l, betaQ, betaR, gamma1, gamma2, theta, mu)   
#
# The iterator implements Algorithm 3 of [1] with constant stepsize (α_n=λ) for several prominant special cases:
# 1) θ = 2 	 	==>   Corresponds to the Vu-Condat Algorithm [2,3].
# 2) θ = 1, μ=1
# 3) θ = 0, μ=1
# 4) θ ∈ [0,∞), μ=0 
#
# See [1, Figure 1] for other special cases and relation to other algorithms.   
#
# [1] Latafat, Patrinos. "Asymmetric forward–backward–adjoint splitting for solving monotone inclusions involving three operators"  Computational Optimization and Applications, pages 1–37, 2017.
# [2] Condat. "A primal–dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms" Journal of Optimization Theory and Applications 158.2 (2013): 460-479.
# [3] Vũ. "A splitting algorithm for dual monotone inclusions involving cocoercive operators"" Advances in Computational Mathematics, 38(3), pp.667-681.

struct AFBAIterator{I <: Integer, R <: Real, T1 <: BlockArray, T2 <: BlockArray} <: ProximalAlgorithm{I, T1}
    x::T1
    y::T2
    g
    h
    f
    l
    L
    theta::R
    mu::R
    lam::R
    betaQ::R # Lipschitz constant of f
    betaR::R # Lipschitz constant of l
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

function AFBAIterator(x0::T1, y0::T2; g=IndFree(), h=IndFree(), f=IndFree(), l=IndZero(), L=Identity(blocksize(x0)), theta=1, mu=1, lam=1, betaQ=0, betaR=0, gamma1=-1, gamma2=-1, maxit::I=10000, tol::R=1e-5, verbose=1, verbose_freq = 100) where {I, R, T1, T2}
    x = blockcopy(x0)
    xbar = blockcopy(x0)
    y = blockcopy(y0)
    ybar = blockcopy(y0)
    gradf = blockcopy(x0)
    gradl = blockcopy(y0)
    FPR_x = blockcopy(x0)
    FPR_x .= Inf
    FPR_y = blockcopy(y0)
    FPR_y .= Inf
	temp_x = blockcopy(x0)
	temp_y = blockcopy(y0)
    hconj = Conjugate(h) 
    lconj = Conjugate(l)

	# default stepsizes
    par= 4; #  scale parameter for comparing Lipschitz constant and norm(L)
    nmL=norm(L)
    alpha=1;
    if isa(h,ProximalOperators.IndFree) &&  (gamma1<0 ||  gamma2<0) 
    	# mu=0 in this case is the only case where stepsizes matter
    	alpha =1000/(betaQ+1e-5) # the speed is determined by gamma1 since bary=0
    	temp = theta^2-3*theta+3 
		gamma1 = 0.99/(betaQ/2+ temp*nmL/alpha);  # in this case R=0
        gamma2 = 1/(nmL*alpha);
    else 
    if gamma1<0 || gamma2<0
        if theta==2 #default stepsize for Vu-Condat
            if betaQ > par*nmL && betaR > par*nmL
                alpha = 1;
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ;
            elseif  betaR > par*nmL
                alpha = betaR/(2*nmL);
            end    
            gamma1 = 1/(betaQ/2+nmL/alpha); 
            gamma2 = 0.99/(betaR/2+nmL*alpha); 
        elseif theta==1 && mu==1 # default stepsize for theta=1, mu=1
            if betaQ > par*nmL && betaR > par*nmL
                alpha = 1;
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ;
            elseif  betaR > par*nmL
                alpha = betaR/(2*nmL);
            end    
            gamma1 = 1/(betaQ/2+nmL/alpha); 
            gamma2 = 0.99/(betaR/2+gamma1*nmL^2); 
        elseif theta==0 && mu==1 # default stepsize for theta=0, mu=1
            temp=3;
            if betaQ > par*nmL && betaR > temp*par*nmL # denominator for Sigma involves 3α norm(L) in this case
                alpha = 1;
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ;
            elseif  betaR > temp*par*nmL # denominator for Sigma involves 3α norm(L) in this case
                alpha = betaR/(temp*2*nmL);
            end    
            gamma1 = 1/(betaQ/2+nmL/alpha); 
            gamma2 = 0.99/(betaR/2+2*gamma1*nmL^2+ alpha*nmL); 
        elseif mu==0 # default stepsize for  mu=0
            temp = theta^2-3*theta+3
            if betaQ > temp*par*nmL && betaR > par*nmL
                alpha = 1;
            elseif  betaR > par*nmL 
                alpha = betaR/(2*nmL);
            elseif betaQ > temp*par*nmL # denominator for Sigma involves 3α norm(L) in this case
                alpha = 2*nmL*temp/betaQ;
            end     
            gamma2 = 1/(betaR/2+ alpha*nmL);
            gamma1 = 0.99/(betaQ/2+ (temp-1)*gamma2*nmL^2+ nmL/alpha); 
        else 
            error("this choice of theta and mu is not supported!")
        end 
    end
    end
    return AFBAIterator{I, R, T1, T2}(x0, y0, g, hconj, f, lconj, L, theta, mu, lam, betaQ, betaR, gamma1, gamma2, maxit, tol, verbose, verbose_freq, xbar, ybar, gradf, gradl, FPR_x,FPR_y, temp_x, temp_y)
end

################################################################################
# Utility methods
maxit(sol::AFBAIterator) = sol.maxit

converged(sol::AFBAIterator, it) = vecnorm(sol.FPR_x)+vecnorm(sol.FPR_y) <= sol.tol

verbose(sol::AFBAIterator) = sol.verbose > 0
verbose(sol::AFBAIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::AFBAIterator)
    @printf("%6s | %22s | %10s |\n ", "it", "gamma", "fpr")
    @printf("------|------------------------|------------|\n")
end

function display(sol::AFBAIterator, it)
    @printf("%6d | %7.4e, %7.4e | %7.4e |\n", it, sol.gamma1, sol.gamma2, vecnorm(sol.FPR_x)+vecnorm(sol.FPR_y))
end

function Base.show(io::IO, sol::AFBAIterator)
    println(io, "Asymmetric Forward-Backward-Adjoint Splitting" )
    if sol.theta==2
        println(io, "theta               : $(sol.theta)" )
    else 
        println(io, "theta, mu           : $(sol.theta), $(sol.mu)" )
    end
    println(io, "fpr                 : $(vecnorm(sol.FPR_x)+vecnorm(sol.FPR_y))")
    print(  io, "gamma1, gamma2      : $(sol.gamma1), $(sol.gamma2)")
end

################################################################################
# Initialization

function initialize(sol::AFBAIterator)
    return
end

################################################################################
# Iteration

function iterate(sol::AFBAIterator{I, R,T1, T2}, it::I) where {I, R, T1, T2}
		# perform xbar-update step
        gradient!(sol.gradf,sol.f, sol.x)
        Ac_mul_B!(sol.temp_x, sol.L, sol.y)
        sol.temp_x .+=  sol.gradf
        sol.temp_x .*= -sol.gamma1 
        sol.temp_x .+=  sol.x
        prox!(sol.xbar, sol.g, sol.temp_x, sol.gamma1)
    	# perform ybar-update step 
        gradient!(sol.gradl,sol.l, sol.y)    # sol.l= l^*
        sol.temp_x .=  (sol.theta * sol.xbar) .+ ((1-sol.theta) * sol.x) 
        A_mul_B!(sol.temp_y, sol.L, sol.temp_x)
        sol.temp_y .-= sol.gradl
        sol.temp_y .*= sol.gamma2
        sol.temp_y .+=  sol.y
        prox!(sol.ybar, sol.h, sol.temp_y, sol.gamma2) # sol.h= h^* 
        # the residues
        sol.FPR_x .= sol.xbar .- sol.x   
        sol.FPR_y .= sol.ybar .- sol.y   
        # perform x-update step
        sol.temp_y .= sol.mu*(2-sol.theta)*sol.gamma1*sol.FPR_y
		Ac_mul_B!(sol.temp_x, sol.L, sol.temp_y)
        sol.x .+=  sol.lam *(sol.FPR_x .- sol.temp_x)
        # perform y-update step
        sol.temp_x .= (1-sol.mu)*(2-sol.theta)*sol.gamma2*sol.FPR_x
		A_mul_B!(sol.temp_y, sol.L, sol.temp_x)
        sol.y .+=  sol.lam *(sol.FPR_y .+ sol.temp_y)
    return sol.x
end

################################################################################
# Solver interface(s)

function AFBA(x0,y0; kwargs...)
    # Create iterable
    sol = AFBAIterator(x0,y0; kwargs...)
    # Run iterations
    (it, point) = run(sol)
    return (it, point, sol)
end
