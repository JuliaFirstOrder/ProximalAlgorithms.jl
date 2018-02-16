################################################################################
# ZeroFPR iterator (with L-BFGS directions)

mutable struct ZeroFPRIterator{I <: Integer, R <: Real, T <: BlockArray} <: ProximalAlgorithm{I,T}
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
    xbarbar # proximal-gradient step
    xnew 
    xnewbar 
    H # inverse Jacobian approximation
    FPR_x
    Aqx
    Asx
    Aqxbar
    Asxbar
    Aqxnew
    Asxnew
    Aqd
    Asd
    gradfq_Aqx
    gradfs_Asx
    fs_Asx
    fq_Aqx
    f_Ax
    At_gradf_Ax
    Ast_gradfs_Asx
    Aqt_gradfq_Aqx
    g_xbar
    FBE_x
    FPR_xbar
    FPR_xbar_prev
    d
end

################################################################################
# Constructor

function ZeroFPRIterator(x0::T; fs=Zero(), As=Identity(blocksize(x0)), fq=Zero(), Aq=Identity(blocksize(x0)), g=Zero(), gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, adaptive=false, memory=10, verbose=1, verbose_freq=100, alpha=0.95, sigma=0.5) where {I, R, T}
    n = blocksize(x0)
    mq = size(Aq, 1)
    ms = size(As, 1)
    x = blockcopy(x0)
    y = blockzeros(x0)
    xbar = blockzeros(x0)
    xbarbar = blockzeros(x0)
    xnew = blockzeros(x0)
    xnewbar = blockzeros(x0)
    xbar = blockzeros(x0)
    FPR_x = blockzeros(x0)
    FPR_xbar_prev = blockzeros(x0)
    FPR_xbar = blockzeros(x0)
    Aqx = blockzeros(mq)
    Asx = blockzeros(ms)
    Aqxbar = blockzeros(mq)
    Asxbar = blockzeros(ms)
    Aqxnew = blockzeros(mq)
    Asxnew = blockzeros(ms)
    Aqd = blockzeros(mq)
    Asd = blockzeros(ms)
    gradfq_Aqx = blockzeros(mq)
    gradfs_Asx = blockzeros(ms)
    At_gradf_Ax = blockzeros(n)
    Ast_gradfs_Asx = blockzeros(n)
    Aqt_gradfq_Aqx = blockzeros(n)
    d = blockzeros(x0)
    H = LBFGS(x, memory)
    ZeroFPRIterator{I, R, T}(x, 
			     fs, As, 
			     fq, Aq, g, 
			     gamma, maxit, tol, adaptive, verbose, verbose_freq, 
			     alpha, sigma, 0.0, y, 
			     xbar, xbarbar, xnew, xnewbar, 
			     H, FPR_x, 
			     Aqx, Asx, 
			     Aqxbar, Asxbar, 
			     Aqxnew, Asxnew, 
			     Aqd, Asd, 
			     gradfq_Aqx, gradfs_Asx, 
			     0.0, 0.0, 0.0, 
			     At_gradf_Ax, Ast_gradfs_Asx, Aqt_gradfq_Aqx,
			     0.0, 0.0, 
			     FPR_xbar, FPR_xbar_prev, d)
end

################################################################################
# Utility methods

maxit(sol::ZeroFPRIterator) = sol.maxit

converged(sol::ZeroFPRIterator, it) = it > 0 && blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::ZeroFPRIterator) = sol.verbose > 0 
verbose(sol::ZeroFPRIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::ZeroFPRIterator)
	@printf("%6s | %10s | %10s | %10s | %10s |\n ", "it", "gamma", "fpr", "tau", "FBE")
	@printf("------|------------|------------|------------|------------|\n")
end
function display(sol::ZeroFPRIterator, it)
    @printf("%6d | %7.4e | %7.4e | %7.4e | %7.4e | \n", it, sol.gamma, blockmaxabs(sol.FPR_x)/sol.gamma, sol.tau, sol.FBE_x)
end

function Base.show(io::IO, sol::ZeroFPRIterator)
	println(io, "ZeroFPR" )
	println(io, "fpr        : $(blockmaxabs(sol.FPR_x))")
	println(io, "gamma      : $(sol.gamma)")
	println(io, "tau        : $(sol.tau)")
	print(  io, "FBE        : $(sol.FBE_x)")
end

################################################################################
# Initialization

function initialize!(sol::ZeroFPRIterator)

    # reset L-BFGS operator (would be nice to have this option)
    # TODO add function reset!(::LBFGS) in AbstractOperators
    sol.H.currmem, sol.H.curridx = 0, 0
    sol.H.H = 1.0

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

end

################################################################################
# Iteration

function iterate!(sol::ZeroFPRIterator{I, R, T}, it::I) where {I, R, T}

    # These need to be performed anyway (to compute xbarbar later on)
    A_mul_B!(sol.Aqxbar, sol.Aq, sol.xbar)
    fq_Aqxbar = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxbar)
    A_mul_B!(sol.Asxbar, sol.As, sol.xbar)
    fs_Asxbar = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxbar)
    f_Axbar = fs_Asxbar + fq_Aqxbar

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            normFPR_x = blockvecnorm(sol.FPR_x)
            uppbnd = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_x^2
            if f_Axbar > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
                sol.xbar, sol.g_xbar = prox(sol.g, sol.y, sol.gamma)
                blockaxpy!(sol.FPR_x, sol.x, -1.0, sol.xbar)
            else
                break
            end
	    A_mul_B!(sol.Aqxbar, sol.Aq, sol.xbar)
	    fq_Aqxbar = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxbar)
	    A_mul_B!(sol.Asxbar, sol.As, sol.xbar)
	    fs_Asxbar = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxbar)
	    f_Axbar = fs_Asxbar + fq_Aqxbar
        end
    end

    # Compute value of FBE at x

    normFPR_x = blockvecnorm(sol.FPR_x)
    FBE_x = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_x^2 + sol.g_xbar

    # Compute search direction
    Ac_mul_B!(sol.Ast_gradfs_Asx, sol.As, sol.gradfs_Asx)
    Ac_mul_B!(sol.Aqt_gradfq_Aqx, sol.Aq, sol.gradfq_Aqx)
    blockaxpy!(sol.At_gradf_Ax, sol.Ast_gradfs_Asx, 1.0, sol.Aqt_gradfq_Aqx)
    blockaxpy!(sol.y, sol.xbar, -sol.gamma, sol.At_gradf_Ax)
    g_xbarbar = prox!(sol.xbarbar, sol.g, sol.y, sol.gamma)
    blockaxpy!(sol.FPR_xbar, sol.xbar, -1.0, sol.xbarbar)

    if it > 1
        update!(sol.H, sol.xbar, sol.xnewbar, sol.FPR_xbar, sol.FPR_xbar_prev)
    end
    A_mul_B!(sol.d, sol.H, 0.0 .- sol.FPR_xbar) # TODO: not nice

    # Perform line-search over the FBE

    sol.tau = 1.0

    A_mul_B!(sol.Asd, sol.As, sol.d)
    A_mul_B!(sol.Aqd, sol.Aq, sol.d)

    C = sol.sigma*sol.gamma*(1.0-sol.alpha)

    g_xnewbar = zero(R)
    f_Axnew = zero(R)
    FBE_xnew = zero(R)

    maxit_tau = 10
    for it_tau = 1:maxit_tau # TODO: replace/complement with lower bound on tau
	blockaxpy!(sol.xnew, sol.xbar, sol.tau, sol.d)
	blockaxpy!(sol.Asxnew, sol.Asxbar, sol.tau, sol.Asd)
	blockaxpy!(sol.Aqxnew, sol.Aqxbar, sol.tau, sol.Aqd)
        fs_Asxnew = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxnew)
        # TODO: can precompute most of next line before the iteration
        fq_Aqxnew = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxnew)
        f_Axnew = fs_Asxnew + fq_Aqxnew
	Ac_mul_B!(sol.Ast_gradfs_Asx, sol.As, sol.gradfs_Asx)
	Ac_mul_B!(sol.Aqt_gradfq_Aqx, sol.Aq, sol.gradfq_Aqx)
	blockaxpy!(sol.At_gradf_Ax, sol.Ast_gradfs_Asx, 1.0, sol.Aqt_gradfq_Aqx)
	blockaxpy!(sol.y, sol.xnew, -sol.gamma, sol.At_gradf_Ax)
        g_xnewbar = prox!(sol.xnewbar, sol.g, sol.y, sol.gamma)
	blockaxpy!(sol.FPR_x, sol.xnew, -1.0, sol.xnewbar)
        normFPR_xnew = blockvecnorm(sol.FPR_x)
        FBE_xnew = f_Axnew - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_xnew^2 + g_xnewbar
        if FBE_xnew <= FBE_x - (C/2)*normFPR_x^2
            break
        end
        sol.tau = 0.5*sol.tau
    end

    sol.FBE_x = FBE_xnew
    sol.x, sol.xnew = sol.xnew, sol.x
    sol.f_Ax = f_Axnew
    sol.g_xbar = g_xnewbar
    sol.FPR_xbar_prev, sol.FPR_xbar = sol.FPR_xbar, sol.FPR_xbar_prev
    sol.xbar, sol.xnewbar = sol.xnewbar, sol.xbar #xnewbar becames xbar_prev

    return sol.xbar 

end

################################################################################
# Solver interface(s)

function ZeroFPR(x0; kwargs...)
    sol = ZeroFPRIterator(x0; kwargs...)
    it, point = run!(sol)
    return (it, point, sol)
end
