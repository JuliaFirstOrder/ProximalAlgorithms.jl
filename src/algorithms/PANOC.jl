################################################################################
# PANOC iterator (with L-BFGS directions)

mutable struct PANOCIterator{I <: Integer, R <: Real, T <: BlockArray} <: ProximalAlgorithm{I,T}
    x::T
    fs
    As
    fq
    Aq
    g
    gamma::R
    maxit::I
    tol::R
    adaptive::Bool
    verbose::I
    verbose_freq::I
    alpha::R
    sigma::R
    tau::R
    y # gradient step
    xbar # proximal-gradient step
    H # inverse Jacobian approximation
    FPR_x
    Aqx
    Asx
    gradfq_Aqx
    gradfs_Asx
    fs_Asx
    fq_Aqx
    f_Ax
    At_gradf_Ax
    g_xbar
    FBE_x
    FBE_x_prev
    x_prev
    FPR_x_prev
    d
    normFPR_x
end

################################################################################
# Constructor

function PANOCIterator(x0::T; fs=Zero(), As=Identity(blocksize(x0)), fq=Zero(), Aq=Identity(blocksize(x0)), g=Zero(), gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, adaptive=false, memory=10, verbose=1, verbose_freq=100, alpha=0.95, sigma=0.5) where {I, R, T}
    n = blocksize(x0)
    mq = size(Aq, 1)
    ms = size(As, 1)
    x = blockcopy(x0)
    x_prev = blockcopy(x0)
    y = blockzeros(x0)
    xbar = blockzeros(x0)
    FPR_x = blockzeros(x0)
    FPR_x_prev = blockzeros(x0)
    Aqx = blockzeros(mq)
    Asx = blockzeros(ms)
    gradfq_Aqx = blockzeros(mq)
    gradfs_Asx = blockzeros(ms)
    At_gradf_Ax = blockzeros(n)
    d = blockzeros(x0)
    PANOCIterator{I, R, T}(x, fs, As, fq, Aq, g, gamma, maxit, tol, adaptive, verbose, verbose_freq, alpha, sigma, 0.0, y, xbar, LBFGS(x, memory), FPR_x, Aqx, Asx, gradfq_Aqx, gradfs_Asx, 0.0, 0.0, 0.0, At_gradf_Ax, 0.0, 0.0, 0.0, x_prev, FPR_x_prev, d, 0)
end

################################################################################
# Utility methods

maxit(sol::PANOCIterator) = sol.maxit

converged(sol::PANOCIterator, it) = it > 0 && blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::PANOCIterator) = sol.verbose > 0 
verbose(sol::PANOCIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::PANOCIterator)
	@printf("%6s | %10s | %10s | %10s | %10s |\n ", "it", "gamma", "fpr", "tau", "FBE")
	@printf("------|------------|------------|------------|------------|\n")
end
function display(sol::PANOCIterator, it)
    @printf("%6d | %7.4e | %7.4e | %7.4e | %7.4e | \n", it, sol.gamma, blockmaxabs(sol.FPR_x)/sol.gamma, sol.tau, sol.FBE_x)
end

function Base.show(io::IO, sol::PANOCIterator)
	println(io, "PANOC" )
	println(io, "fpr        : $(blockmaxabs(sol.FPR_x))")
	println(io, "gamma      : $(sol.gamma)")
	println(io, "tau        : $(sol.tau)")
	print(  io, "FBE        : $(sol.FBE_x)")
end

################################################################################
# Initialization

function initialize!(sol::PANOCIterator)

    # compute first forward-backward step here
    A_mul_B!(sol.Aqx, sol.Aq, sol.x)
    sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
    A_mul_B!(sol.Asx, sol.As, sol.x)
    sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
    blockaxpy!(sol.At_gradf_Ax, sol.As'*sol.gradfs_Asx, 1.0, sol.Aq'*sol.gradfq_Aqx)
    sol.f_Ax = sol.fs_Asx + sol.fq_Aqx

    if sol.gamma <= 0.0 # estimate L in this case, and set gamma = 1/L
        # this part should be as follows:
        # 1) if adaptive = false and only fq is present then L is "accurate"
        # 2) otherwise L is "inaccurate" and set adaptive = true
        # TODO: implement case 1), now 2) is always performed
        xeps = sol.x .+ sqrt(eps())
        Aqxeps = sol.Aq*xeps
        gradfq_Aqxeps, = gradient(sol.fq, Aqxeps)
        Asxeps = sol.As*xeps
        gradfs_Asxeps, = gradient(sol.fs, Asxeps)
        At_gradf_Axeps = sol.As'*gradfs_Asxeps .+ sol.Aq'*gradfq_Aqxeps
        L = blockvecnorm(sol.At_gradf_Ax .- At_gradf_Axeps)/(sqrt(eps()*blocklength(xeps)))
        sol.adaptive = true
        # in both cases set gamma = 1/L
        sol.gamma = sol.alpha/L
    end

    blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
    sol.g_xbar = prox!(sol.xbar, sol.g, sol.y, sol.gamma)
    blockaxpy!(sol.FPR_x, sol.x, -1.0, sol.xbar)

    normFPR_x = blockvecnorm(sol.FPR_x)
    sol.FBE_x = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_x^2 + sol.g_xbar



end

################################################################################
# Iteration

function iterate!(sol::PANOCIterator{I, R, T}, it::I) where {I, R, T}

	if sol.adaptive
		for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
			Aqxbar = sol.Aq*sol.xbar
			gradfq_Aqxbar, fq_Aqxbar = gradient(sol.fq, Aqxbar)
			Asxbar = sol.As*sol.xbar
			gradfs_Asxbar, fs_Asxbar = gradient(sol.fs, Asxbar)
			f_Axbar = fs_Asxbar + fq_Aqxbar

			uppbnd = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 
				 0.5/sol.gamma*sol.normFPR_x^2
			if f_Axbar > uppbnd + 1e-6*abs(sol.f_Ax)
				sol.gamma = 0.5*sol.gamma
				blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
				sol.g_xbar = prox!(sol.xbar, sol.g, sol.y, sol.gamma)
				blockaxpy!(sol.FPR_x, sol.x, -1.0, sol.xbar)
			else
				break
			end
			sol.normFPR_x = blockvecnorm(sol.FPR_x)
			sol.FBE_x = uppbnd + sol.g_xbar
		end
	end

	xbar = sol.xbar
	x = sol.x

	if it > 1
		update!(sol.H, sol.x, sol.x_prev, sol.FPR_x, sol.FPR_x_prev)
	end
	A_mul_B!(sol.d, sol.H, 0.0 .- sol.FPR_x) # TODO: not nice

	sol.FPR_x_prev, sol.FPR_x = sol.FPR_x, sol.FPR_x_prev
	blockset!(sol.x_prev, sol.x)

	FBE_x = sol.FBE_x
	normFPR_x = sol.normFPR_x
	C = sol.sigma*sol.gamma*(1.0-sol.alpha)
	maxit_tau = 10
	tau = 1.0
    
	for it_tau = 1:maxit_tau # TODO: replace/complement with lower bound on tau

		xnew = (1-tau).*xbar .+ tau .* (x.+sol.d)
	   
		# calculate new FBE in xnew

		A_mul_B!(sol.Aqx, sol.Aq, xnew)
		sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
		A_mul_B!(sol.Asx, sol.As, xnew)
		sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
		blockaxpy!(sol.At_gradf_Ax, sol.As'*sol.gradfs_Asx, 1.0, sol.Aq'*sol.gradfq_Aqx)
		sol.f_Ax = sol.fs_Asx + sol.fq_Aqx

		# gradient step
		blockaxpy!(sol.y, xnew, -sol.gamma, sol.At_gradf_Ax)
		# prox step
		xnewbar, g_xnewbar = prox(sol.g, sol.y, sol.gamma)
	    
		FPR_xnew = xnew .- xnewbar
		norm_FPRxnew = blockvecnorm(FPR_xnew)
	    
		FBE_xnew = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, FPR_xnew) + 
		           0.5/sol.gamma*norm_FPRxnew^2 + g_xnewbar

		if FBE_xnew <= FBE_x - (C/2)*normFPR_x^2 || it_tau == maxit_tau
			sol.normFPR_x = norm_FPRxnew
			sol.FPR_x = FPR_xnew
			sol.FBE_x = FBE_xnew
			sol.x, sol.xbar = xnew, xnewbar
			sol.tau = tau
			break
		end
		tau *= 0.5
    
	end

	return sol.xbar 

end

################################################################################
# Solver interface(s)

function PANOC(x0; kwargs...)
    sol = PANOCIterator(x0; kwargs...)
    it, point = run!(sol)
    return (it, point, sol)
end
