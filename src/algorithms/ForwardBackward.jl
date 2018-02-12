################################################################################
# Forward-backward splitting iterator
#
# This iterator implements the algorithms ISTA (when fast=False) and FISTA
# (when fast=True) described in
# [1] Beck, Teboulle "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems," SIAM Journal on Imaging Sciences vol. 2, no. 1, pp. 183-202 (2009)
# or equivalently (when fast=True) Algorithm 2 described in
# [2] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization," (2008)
#

mutable struct FBSIterator{I <: Integer, R <: Real, T <: BlockArray{R}} <: ProximalAlgorithm{I}
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
    fast::Bool
    verbose::I
    verbose_freq::I
    theta::R # extrapolation parameter
    y # gradient step
    z # proximal-gradient step
    z_prev
    FPR_x
    Aqx
    Asx
    gradfq_Aqx
    gradfs_Asx
    fs_Asx
    fq_Aqx
    f_Ax
    At_gradf_Ax
    Aqz_prev
    Asz_prev
    gradfq_Aqz_prev
end

################################################################################
# Constructor

function FBSIterator(x0::T; fs=Zero(), As=Identity(blocksize(x0)), fq=Zero(), Aq=Identity(blocksize(x0)), g=Zero(), gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, adaptive=false, fast=false, verbose=1, verbose_freq = 100) where {I, R, T}
    n = blocksize(x0)
    mq = size(Aq, 1)
    ms = size(As, 1)
    x = blockcopy(x0)
    y = blockzeros(x0)
    z = blockzeros(x0)
    z_prev = blockzeros(x0)
    FPR_x = blockzeros(x0)
    Aqx = blockzeros(mq)
    Asx = blockzeros(ms)
    gradfq_Aqx = blockzeros(mq)
    gradfs_Asx = blockzeros(ms)
    At_gradf_Ax = blockzeros(n)
    Aqz_prev = blockzeros(mq)
    Asz_prev = blockzeros(ms)
    gradfq_Aqz_prev = blockzeros(mq)
    FBSIterator{I, R, T}(x, fs, As, fq, Aq, g, gamma, maxit, tol, adaptive, fast, verbose, verbose_freq, 1.0, y, z, z_prev, FPR_x, Aqx, Asx, gradfq_Aqx, gradfs_Asx, 0.0, 0.0, 0.0, At_gradf_Ax, Aqz_prev, Asz_prev, gradfq_Aqz_prev)
end

################################################################################
# Utility methods

maxit(sol::FBSIterator) = sol.maxit

converged(sol::FBSIterator, it) = blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::FBSIterator) = sol.verbose > 0
verbose(sol::FBSIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::FBSIterator)
	@printf("%6s | %10s | %10s |\n ", "it", "gamma", "fpr")
	@printf("------|------------|------------|\n")
end

function display(sol::FBSIterator, it)
	@printf("%6d | %7.4e | %7.4e |\n", it, sol.gamma, blockmaxabs(sol.FPR_x)/sol.gamma)
end

function Base.show(io::IO, sol::FBSIterator)
	println(io, (sol.fast ? "Fast " : "")*"Forward-Backward Splitting" )
	println(io, "fpr        : $(blockmaxabs(sol.FPR_x))")
	print(  io, "gamma      : $(sol.gamma)")
end

################################################################################
# Initialization

function initialize!(sol::FBSIterator)

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
        sol.gamma = 1.0/L
    end

    blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
    prox!(sol.z, sol.g, sol.y, sol.gamma)
    blockaxpy!(sol.FPR_x, sol.x, -1.0, sol.z)

end

################################################################################
# Iteration

function iterate!(sol::FBSIterator{I, R, T}, it::I) where {I, R, T}

    Aqz = 0.0
    gradfq_Aqz = 0.0
    fq_Aqz = 0.0
    Asz = 0.0
    gradfs_Asz = 0.0
    fs_Asz = 0.0

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            normFPR_x = blockvecnorm(sol.FPR_x)
            uppbnd = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_x^2
            # TODO: we can save allocations in the next four lines
            Aqz = sol.Aq*sol.z
            gradfq_Aqz, fq_Aqz = gradient(sol.fq, Aqz)
            Asz = sol.As*sol.z
            gradfs_Asz, fs_Asz = gradient(sol.fs, Asz)
            f_Az = fs_Asz + fq_Aqz
            if f_Az > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
                prox!(sol.z, sol.g, sol.y, sol.gamma)
                blockaxpy!(sol.FPR_x, sol.x, -1.0, sol.z)
            else
                break
            end
        end
    end

    if sol.fast == false
        sol.x, sol.z = sol.z, sol.x
        if sol.adaptive == true
            sol.Aqx = Aqz
            sol.fq_Aqx = fq_Aqz
            sol.gradfq_Aqx = gradfq_Aqz
            sol.Asx = Asz
            sol.fs_Asx = fs_Asz
            sol.gradfs_Asx = gradfs_Asz
            # TODO: we can save allocations in the next line
            blockaxpy!(sol.At_gradf_Ax, sol.As'*sol.gradfs_Asx, 1.0, sol.Aq'*sol.gradfq_Aqx)
            sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
        end
    else
        # compute extrapolation coefficient
        theta1 = (1.0+sqrt(1.0+4*sol.theta^2))/2.0
        extr = (sol.theta - 1.0)/theta1
        sol.theta = theta1
        # perform extrapolation
        sol.x .= sol.z .+ extr.*(sol.z .- sol.z_prev)
        sol.z, sol.z_prev = sol.z_prev, sol.z
        if sol.adaptive == true
            # extrapolate other extrapolable quantities
            diff_Aqx = extr.*(Aqz .- sol.Aqz_prev)
            sol.Aqx .= Aqz .+ diff_Aqx
            diff_gradfq_Aqx = extr.*(gradfq_Aqz .- sol.gradfq_Aqz_prev)
            sol.gradfq_Aqx .= gradfq_Aqz .+ diff_gradfq_Aqx
            sol.fq_Aqx = fq_Aqz + blockvecdot(gradfq_Aqz, diff_Aqx) + 0.5*blockvecdot(diff_Aqx, diff_gradfq_Aqx)
            sol.Asx .= Asz .+ extr.*(Asz .- sol.Asz_prev)
            # store the z-quantities for future extrapolation
            sol.Aqz_prev = Aqz
            sol.gradfq_Aqz_prev = gradfq_Aqz
            sol.Asz_prev = Asz
            # compute gradient of fs
            sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
            # TODO: we can probably save the MATVEC by Aq' in the next line
            # TODO: we can save allocations in the next line
            blockaxpy!(sol.At_gradf_Ax, sol.As'*sol.gradfs_Asx, 1.0, sol.Aq'*sol.gradfq_Aqx)
            sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
        end
    end
    if sol.adaptive == false
        A_mul_B!(sol.Aqx, sol.Aq, sol.x)
        sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
        A_mul_B!(sol.Asx, sol.As, sol.x)
        sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
        # TODO: we can save allocations in the next line
        blockaxpy!(sol.At_gradf_Ax, sol.As'*sol.gradfs_Asx, 1.0, sol.Aq'*sol.gradfq_Aqx)
        sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
    end
    blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
    prox!(sol.z, sol.g, sol.y, sol.gamma)
    blockaxpy!(sol.FPR_x, sol.x, -1.0, sol.z)

    return 

end

################################################################################
# Solver interface

"""
**Forward-backward splitting**

    FBS(x0; kwargs...)

Solves a problem of the form

    minimize fs(As*x) + fq(Aq*x) + g(x)

where `fs` is a smooth function, `fq` is a quadratic function, `g` is a
proximable function and `As`, `Aq` are linear operators. Parameter `x0` is the
initial point. Keyword arguments specify the problem and
additional options as follows:
* `fs`, smooth function (default: identically zero function)
* `fq`, quadratic function (default: identically zero function)
* `g`, proximable function (default: identically zero function)
* `As`, linear operator (default: identity)
* `Aq`, linear operator (default: identity)
* `gamma`, stepsize (default: unspecified, determine automatically)
* `maxit`, maximum number of iteration (default: `10000`)
* `tol`, halting tolerance on the fixed-point residual (default: `1e-4`)
* `adaptive`, adaptively adjust `gamma` (default: `false` if `gamma` is provided)
* `fast`, enables accelerated method (default: `false`)
* `verbose`, verbosity level (default: `1`)
* `verbose_freq`, verbosity frequency for `verbose = 1` (default: `100`)
"""

function FBS(x0; kwargs...)
    sol = FBSIterator(x0; kwargs...)
    return FBS!(sol)
end

function FBS!(sol::FBSIterator)
    it = run!(sol)
    return (it, sol.z, sol)
end
