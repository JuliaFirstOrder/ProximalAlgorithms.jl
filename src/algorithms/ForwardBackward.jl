################################################################################
# Forward-backward splitting iterator
#
# This iterator implements the algorithms ISTA (when fast=False) and FISTA
# (when fast=True) described in
# [1] Beck, Teboulle "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems," SIAM Journal on Imaging Sciences vol. 2, no. 1, pp. 183-202 (2009)
# or equivalently (when fast=True) Algorithm 2 described in
# [2] Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization," (2008)
#

mutable struct FBSIterator{I <: Integer, R <: Real, D, CS, FS, AS, CQ, FQ, AQ, G} <: ProximalAlgorithm{I, D}
    x::D
    fs::FS
    As::AS
    fq::FQ
    Aq::AQ
    g::G
    gamma::R
    maxit::I
    tol::R
    adaptive::Bool
    fast::Bool
    verbose::I
    verbose_freq::I
    theta::R # extrapolation parameter
    y::D # gradient step
    z::D # proximal-gradient step
    z_prev::D
    FPR_x::D
    Aqx::CQ
    Asx::CS
    Aqz::CQ
    Asz::CS
    gradfq_Aqx::CQ
    gradfs_Asx::CS
    Aqt_gradfq_Aqx::D
    Ast_gradfs_Asx::D
    fs_Asx::R
    fq_Aqx::R
    f_Ax::R
    At_gradf_Ax::D
    Aqz_prev::CQ
    Asz_prev::CS
    gradfq_Aqz::CQ
    gradfs_Asz::CS
    gradfq_Aqz_prev::CQ
end

################################################################################
# Constructor

function FBSIterator(x0::D; 
                     fs::FS=Zero(), As::AS=Identity(size(x0)), 
                     fq::FQ=Zero(), Aq::AQ=Identity(size(x0)), 
                     g::G=Zero(), 
                     gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, 
                     adaptive::Bool=false, fast::Bool=false, 
                     verbose::I=1, verbose_freq::I = 100) where {I, R, D, FS, AS, FQ, AQ, G}
    x = copy(x0)
    y = zero(x0)
    z = zero(x0)
    z_prev = zero(x0)
    FPR_x = zero(x0)
    Aqx = Aq*x
    Asx = As*x
    Aqz = zero(Aqx)
    Asz = zero(Asx)
    gradfq_Aqx = zero(Aqx)
    gradfs_Asx = zero(Asx)
    Aqt_gradfq_Aqx = zero(x0)
    Ast_gradfs_Asx = zero(x0)
    At_gradf_Ax = zero(x0)
    Aqz_prev = zero(Aqx)
    Asz_prev = zero(Asx)
    gradfq_Aqz = zero(Aqx)
    gradfs_Asz = zero(Asx)
    gradfq_Aqz_prev = zero(Aqx)
    CQ = typeof(Aqx)
    CS = typeof(Asx)
    FBSIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G}(x,
             fs, As, fq, Aq, g,
             gamma, maxit, tol, adaptive, fast, verbose, verbose_freq, 1.0,
             y, z, z_prev, FPR_x,
             Aqx, Asx,
             Aqz, Asz,
             gradfq_Aqx, gradfs_Asx,
             Aqt_gradfq_Aqx, Ast_gradfs_Asx,
             0.0, 0.0, 0.0,
             At_gradf_Ax, Aqz_prev,
             Asz_prev,
             gradfq_Aqz, gradfs_Asz, gradfq_Aqz_prev)
end

################################################################################
# Utility methods

maxit(sol::FBSIterator{I}) where {I} = sol.maxit

converged(sol::FBSIterator{I, R, D}, it::I) where {I, R, D}  = it > 0 && maximum(abs,sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::FBSIterator) = sol.verbose > 0
verbose(sol::FBSIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::FBSIterator)
    @printf("%6s | %10s | %10s |\n ", "it", "gamma", "fpr")
    @printf("------|------------|------------|\n")
end

function display(sol::FBSIterator, it)
    @printf("%6d | %7.4e | %7.4e |\n", it, sol.gamma, maximum(abs,sol.FPR_x)/sol.gamma)
end

function Base.show(io::IO, sol::FBSIterator)
    println(io, (sol.fast ? "Fast " : "")*"Forward-Backward Splitting" )
    println(io, "fpr        : $(maximum(abs,sol.FPR_x))")
    print(  io, "gamma      : $(sol.gamma)")
end

################################################################################
# Initialization

function initialize!(sol::FBSIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G}) where {I, R, D, CS, FS, AS, CQ, FQ, AQ, G}

    # reset parameters
    sol.theta = 1.0

    # compute first forward-backward step here
    mul!(sol.Aqx, sol.Aq, sol.x)
    sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
    mul!(sol.Asx, sol.As, sol.x)
    sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
    mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)
    mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
    sol.At_gradf_Ax .= sol.Ast_gradfs_Asx .+ sol.Aqt_gradfq_Aqx
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
        L = norm(sol.At_gradf_Ax .- At_gradf_Axeps)/(sqrt(eps()*length(xeps)))
        sol.adaptive = true
        # in both cases set gamma = 1/L
        sol.gamma = 1.0/L
    end

    sol.y .= sol.x .- sol.gamma .* sol.At_gradf_Ax
    prox!(sol.z, sol.g, sol.y, sol.gamma)
    sol.FPR_x .= sol.x .- sol.z

    return sol.z
end

################################################################################
# Iteration

function iterate!(sol::FBSIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G}, it::I) where {I, R, D, CS, FS, AS, CQ, FQ, AQ, G}

    fq_Aqz = zero(R)
    fs_Asz = zero(R)

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            normFPR_x = norm(sol.FPR_x)
            uppbnd = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_x)) + 0.5/sol.gamma*normFPR_x^2
            mul!(sol.Aqz, sol.Aq, sol.z)
            fq_Aqz = gradient!(sol.gradfq_Aqz, sol.fq, sol.Aqz)
            mul!(sol.Asz, sol.As, sol.z)
            fs_Asz = gradient!(sol.gradfs_Asz, sol.fs, sol.Asz)
            f_Az = fs_Asz + fq_Aqz
            if f_Az > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                sol.y .= sol.x - sol.gamma .* sol.At_gradf_Ax
                prox!(sol.z, sol.g, sol.y, sol.gamma)
                sol.FPR_x .= sol.x .- sol.z
            else
                break
            end
        end
    end

    if sol.fast == false
        sol.x, sol.z = sol.z, sol.x
        if sol.adaptive == true
            sol.Aqx, sol.Aqz = sol.Aqz, sol.Aqx
            sol.fq_Aqx = fq_Aqz
            sol.gradfq_Aqx, sol.gradfq_Aqz = sol.gradfq_Aqz, sol.gradfq_Aqx
            sol.Asx, sol.Asz = sol.Asz, sol.Asx
            sol.fs_Asx = fs_Asz
            sol.gradfs_Asx, sol.gradfs_Asz = sol.gradfs_Asz, sol.gradfs_Asx
            mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)
            mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
            sol.At_gradf_Ax .= sol.Ast_gradfs_Asx .+ sol.Aqt_gradfq_Aqx
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
            sol.Aqx .= sol.Aqz .- sol.Aqz_prev
            sol.gradfq_Aqx .= sol.gradfq_Aqz .- sol.gradfq_Aqz_prev
            coeff_linear = real(dot(sol.gradfq_Aqz, sol.Aqx))
            coeff_quadr = real(dot(sol.Aqx, sol.gradfq_Aqx))
            sol.fq_Aqx = fq_Aqz + extr*coeff_linear + 0.5*coeff_quadr*extr^2
            sol.Aqx .= sol.Aqz .+ extr.*(sol.Aqz .- sol.Aqz_prev)
            sol.gradfq_Aqx .= sol.gradfq_Aqz .+ extr.*(sol.gradfq_Aqz .- sol.gradfq_Aqz_prev)
            sol.Asx .= sol.Asz .+ extr .*(sol.Asz .- sol.Asz_prev)
            # store the z-quantities for future extrapolation
            sol.Aqz_prev, sol.Aqz = sol.Aqz, sol.Aqz_prev
            sol.gradfq_Aqz_prev, sol.gradfq_Aqz = sol.gradfq_Aqz, sol.gradfq_Aqz_prev
            sol.Asz_prev, sol.Asz = sol.Asz, sol.Asz_prev
            # compute gradient of fs
            sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
            mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)
            # TODO: we can probably save the MATVEC by Aq' in the next line
            mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
            sol.At_gradf_Ax .= sol.Ast_gradfs_Asx .+ sol.Aqt_gradfq_Aqx
            sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
        end
    end
    if sol.adaptive == false
        mul!(sol.Aqx, sol.Aq, sol.x)
        sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
        mul!(sol.Asx, sol.As, sol.x)
        sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
        mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)
        mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
        sol.At_gradf_Ax .= sol.Ast_gradfs_Asx .+ sol.Aqt_gradfq_Aqx
        sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
    end
    sol.y .= sol.x .- sol.gamma .* sol.At_gradf_Ax
    prox!(sol.z, sol.g, sol.y, sol.gamma)
    sol.FPR_x .= sol.x .- sol.z

    return sol.z

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
    it, point = run!(sol)
    return (it, point, sol)
end
