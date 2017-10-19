mutable struct FBSSolver <: ProximalAlgorithm
    x::AbstractArray{Float64}
    fs
    As
    fq
    Aq
    g
    gamma::Float64
    maxit::Int64
    tol::Float64
    adaptive::Bool
    fast::Bool
    z
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

# function FBSSolver(x0, fs, As, fq, Aq, g, gam, maxit, tol, adaptive, fast)
function FBSSolver(x0; fs=IndFree(), As=Eye(eltype(x0), size(x0)), fq=IndFree(), Aq=Eye(eltype(x0), size(x0)), g=IndFree(), gamma=-1.0, maxit=10000, tol=1e-4, adaptive=false, fast=false)
    n = size(x0)
    ms = size(As, 1)
    mq = size(Aq, 1)
    x = copy(x0)
    z = similar(x0)
    z_prev = similar(x0)
    FPR_x = similar(x0)
    Aqx = zeros(mq)
    Asx = zeros(ms)
    gradfq_Aqx = zeros(mq)
    gradfs_Asx = zeros(ms)
    At_gradf_Ax = zeros(n)
    Aqz_prev = zeros(mq)
    Asz_prev = zeros(ms)
    gradfq_Aqz_prev = zeros(mq)
    FBSSolver(x, fs, As, fq, Aq, g, gamma, maxit, tol, adaptive, fast, z, z_prev, FPR_x, Aqx, Asx, gradfq_Aqx, gradfs_Asx, 0.0, 0.0, 0.0, At_gradf_Ax, Aqz_prev, Asz_prev, gradfq_Aqz_prev)
end

################################################################################
# Utility methods

maxit(solver::FBSSolver) = solver.maxit

converged(solver::FBSSolver, it) = vecnorm(solver.FPR_x, Inf)/solver.gamma <= solver.tol

verbose(solver::FBSSolver, it) = false

display(it, solver::FBSSolver) = println("$(it) $(solver.gamma) $(vecnorm(solver.FPR_x, Inf)/solver.gamma)")

################################################################################
# Initialization

function initialize(sol::FBSSolver)

    # compute first forward-backward step here
    A_mul_B!(sol.Aqx, sol.Aq, sol.x)
    sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
    A_mul_B!(sol.Asx, sol.As, sol.x)
    sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
    sol.At_gradf_Ax = sol.As'*sol.gradfs_Asx + sol.Aq'*sol.gradfq_Aqx
    sol.f_Ax = sol.fs_Asx + sol.fq_Aqx

    if sol.gamma <= 0.0 # estimate L in this case, and set gamma = 1/L
        # 1) if adaptive = false and only fq is present then L is "accurate"
        # TODO: implement this case
        # 2) otherwise L is "inaccurate" and set adaptive = true
        xeps = sol.x .+ sqrt(eps())
        Aqxeps = sol.Aq*xeps
        gradfq_Aqxeps, = gradient(sol.fq, Aqxeps)
        Asxeps = sol.As*xeps
        gradfs_Asxeps, = gradient(sol.fs, Asxeps)
        At_gradf_Axeps = sol.As'*gradfs_Asxeps + sol.Aq'*gradfq_Aqxeps
        L = vecnorm(sol.At_gradf_Ax .- At_gradf_Axeps)/(sqrt(eps()*length(xeps)))
        sol.adaptive = true
        # in both cases set gamma = 1/L
        sol.gamma = 1.0/L
    end

    prox!(sol.z, sol.g, sol.x - sol.gamma*sol.At_gradf_Ax, sol.gamma)
    sol.FPR_x = sol.x - sol.z

end

################################################################################
# Iteration

function iterate(sol::FBSSolver, it)

    Aqz = 0.0
    gradfq_Aqz = 0.0
    fq_Aqz = 0.0
    Asz = 0.0
    gradfs_Asz = 0.0
    fs_Asz = 0.0

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            normFPR_x = vecnorm(sol.FPR_x)
            uppbnd = sol.f_Ax - vecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_x^2;
            Aqz = sol.Aq*sol.z
            gradfq_Aqz, fq_Aqz = gradient(sol.fq, Aqz)
            Asz = sol.As*sol.z
            gradfs_Asz, fs_Asz = gradient(sol.fs, Asz)
            f_Az = fs_Asz + fq_Aqz
            if f_Az > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                sol.z, = prox(sol.g, sol.x - sol.gamma*sol.At_gradf_Ax, sol.gamma)
                sol.FPR_x = sol.x - sol.z
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
            sol.At_gradf_Ax = sol.As'*sol.gradfs_Asx + sol.Aq'*sol.gradfq_Aqx
            sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
        end
        prox!(sol.z, sol.g, sol.x - sol.gamma*sol.At_gradf_Ax, sol.gamma)
        sol.FPR_x = sol.x - sol.z
    else
        extr = it/(it+3)
        sol.x = sol.z + extr*(sol.z - sol.z_prev)
        sol.z, sol.z_prev = sol.z_prev, sol.z
        if sol.adaptive == true
            # extrapolate other extrapolable quantities
            diff_Aqx = extr*(Aqz .- sol.Aqz_prev)
            sol.Aqx = Aqz .+ diff_Aqx
            diff_gradfq_Aqx = extr*(gradfq_Aqz .- sol.gradfq_Aqz_prev)
            sol.gradfq_Aqx = gradfq_Aqz .+ diff_gradfq_Aqx
            sol.fq_Aqx = fq_Aqz + vecdot(gradfq_Aqz, diff_Aqx) + 0.5*vecdot(diff_Aqx, diff_gradfq_Aqx)
            sol.Asx = Asz .+ extr*(Asz .- sol.Asz_prev)
            # store the z-quantities for future extrapolation
            sol.Aqz_prev = Aqz
            sol.gradfq_Aqz_prev = gradfq_Aqz
            sol.Asz_prev = Asz
            # compute gradient of fs
            sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
            # TODO: we can probably save the MATVEC by Aq' in the next line
            sol.At_gradf_Ax = sol.As'*sol.gradfs_Asx + sol.Aq'*sol.gradfq_Aqx
            sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
        end
    end
    if sol.adaptive == false
        A_mul_B!(sol.Aqx, sol.Aq, sol.x)
        sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
        A_mul_B!(sol.Asx, sol.As, sol.x)
        sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
        sol.At_gradf_Ax = sol.As'*sol.gradfs_Asx + sol.Aq'*sol.gradfq_Aqx
        sol.f_Ax = sol.fs_Asx + sol.fq_Aqx
    end
    prox!(sol.z, sol.g, sol.x - sol.gamma*sol.At_gradf_Ax, sol.gamma)
    sol.FPR_x = sol.x - sol.z

end

################################################################################
# Solver interface

function fbs!(x0; kwargs...)
    # create iterable
    solver = FBSSolver(x0; kwargs...)
    # run iterations
    it = run(solver)
    x0 .= solver.x
    return (solver, it)
end

pg! = fbs!
