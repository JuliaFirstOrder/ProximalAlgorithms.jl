################################################################################
# Template iterator

struct AFBAIterator{I <: Integer, R <: Real, T <: BlockArray} <: ProximalAlgorithm{I, T}
    x::T
    y::T
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
    xbar::T
    ybar::T
    gradf::T
    gradl::T
    FPR_x::T
    FPR_y::T
end

################################################################################
# Constructor(s)

function AFBAIterator(x0::T, y0::T; g=Zero(), h=Zero(), f=Zero(), l=Zero(), L=identity(), theta=2, mu=1, lam=1, betaQ=0, betaR=0, gamma1=-1, gamma2=-1, maxit::I=10000, tol::R=1e-5, verbose=1, verbose_freq = 100) where {I, R, T}
# blockcopy can be found in AbstractOperators.jl/src/utilities/block.jl
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

## default stepsizes
    par= 4; #  scale parameter 
    nmL = norm(L);
    alpha=1.0;
    println(" $betaQ, $betaR, $nmL ")    
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
            if betaQ > par*nmL && betaR > temp*par*nmL # denominator for Sigma involves 3alpha norm(L) in this case
                alpha = 1;
            elseif betaQ > par*nmL
                alpha = 2*nmL/betaQ;
            elseif  betaR > temp*par*nmL # denominator for Sigma involves 3alpha norm(L) in this case
                alpha = betaR/(temp*2*nmL);
            end    
            gamma1 = 1/(betaQ/2+nmL/alpha); 
            gamma2 = 0.99/(betaR/2+2*gamma1*nmL^2+ alpha*nmL); 
        elseif mu==0 # default stepsize for  mu=0
            temp = theta^2-3*theta+3
            if betaQ > temp*par*nmL && betaR > par*nmL
                alpha = 1;
            elseif  betaR > par*nmL # denominator for Sigma involves 3alpha norm(L) in this case
                alpha = betaR/(2*nmL);
            elseif betaQ > temp*par*nmL
                alpha = 2*nmL*temp/betaQ;
            end     
            gamma2 = 1/(betaR/2+ alpha*nmL);
            gamma1 = 0.9/(betaQ/2+ (temp-1)*gamma2*nmL^2+ nmL/alpha); 
        else 
            error("this choice of theta and mu is not supported!")
        end 
    end
    return AFBAIterator{I, R, T}(x0, y0, g, h, f, l, L, theta, mu, lam, betaQ, betaR, gamma1, gamma2, maxit, tol, verbose, verbose_freq, xbar, ybar, gradf, gradl, FPR_x,FPR_y)
end

################################################################################
# Utility methods
maxit(sol::AFBAIterator) = sol.maxit

converged(sol::AFBAIterator, it) = norm(sol.FPR_x)+norm(sol.FPR_y) <= sol.tol

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
    println(io, "fpr        : $(norm(sol.FPR_x)+norm(sol.FPR_y))")
    print(  io, "gamma1,gamma2      : $(sol.gamma1), $(sol.gamma2)")
end

################################################################################
# Initialization

function initialize(sol::AFBAIterator)
    return
end

################################################################################
# Iteration

function iterate(sol::AFBAIterator{I, R,T}, it::I) where {I, R, T}
     # perform x-update step
    gradient!(sol.gradf,sol.f, sol.x)
    prox!(sol.xbar, sol.g, sol.x - sol.gamma1*sol.L'*sol.y-sol.gamma1*sol.gradf, sol.gamma1)
    # perform y-update step
    gradient!(sol.gradl,sol.l, sol.y)
    prox!(sol.ybar, Conjugate(sol.h), sol.y + sol.gamma2*sol.L*(sol.theta*sol.xbar+(1-sol.theta)*sol.x)-sol.gamma2*sol.gradl, sol.gamma2)
    sol.FPR_x .= sol.xbar .- sol.x   # fix this later
    sol.FPR_y .= sol.ybar .- sol.y   # fix this later
    sol.x .= sol.x .+ sol.lam .*(sol.FPR_x .- sol.mu*(2-sol.theta)*sol.gamma1*sol.L'*sol.FPR_y) 
    sol.y .= sol.y .+ sol.lam.*(sol.FPR_y .+ (1-sol.mu)*(2-sol.theta)*sol.gamma2*sol.L*sol.FPR_x)
    return sol.x
end

################################################################################
# Solver interface(s)

function AFBA(x0,y0; kwargs...)
    # Create iterable
    # println(typeof(gamma1))
    sol = AFBAIterator(x0,y0; kwargs...)
    println(typeof(sol.gamma1))
    # Run iterations
    (it, point) = run(sol)
    return (it, point, sol)
end
