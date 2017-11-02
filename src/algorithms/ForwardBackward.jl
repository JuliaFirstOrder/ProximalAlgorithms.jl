################################################################################
# Forward-backward splitting iterator

mutable struct FBSIterator{T <: Union{Tuple, AbstractArray}} <: ProximalAlgorithm
    x::T
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

function FBSIterator(x0; fs=Zero(), As=Identity(blocksize(x0)), fq=Zero(), Aq=Identity(blocksize(x0)),  g=Zero(), gamma=-1.0, maxit=10000, tol=1e-4, adaptive=false, fast=false)
    n = blocksize(x0)
    mq = size(Aq, 1)
    ms = size(As, 1)
    x = blockcopy(x0)
    y = blocksimilar(x0)
    z = blocksimilar(x0)
    z_prev = blocksimilar(x0)
    FPR_x = blocksimilar(x0)
    Aqx = blockzeros(mq)
    Asx = blockzeros(ms)
    gradfq_Aqx = blockzeros(mq)
    gradfs_Asx = blockzeros(ms)
    At_gradf_Ax = blockzeros(n)
    Aqz_prev = blockzeros(mq)
    Asz_prev = blockzeros(ms)
    gradfq_Aqz_prev = blockzeros(mq)
    FBSIterator(x, fs, As, fq, Aq, g, gamma, maxit, tol, adaptive, fast, y, z, z_prev, FPR_x, Aqx, Asx, gradfq_Aqx, gradfs_Asx, 0.0, 0.0, 0.0, At_gradf_Ax, Aqz_prev, Asz_prev, gradfq_Aqz_prev)
end

################################################################################
# Utility methods

maxit(sol::FBSIterator) = sol.maxit

converged(sol::FBSIterator, it) = blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::FBSIterator, it) = false

display(it, sol::FBSIterator) = println("$(it) $(sol.gamma) $(blockmaxabs(sol.FPR_x)/sol.gamma)")

################################################################################
# Initialization

function initialize(sol::FBSIterator)

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

function iterate(sol::FBSIterator, it)

    Aqz = 0.0
    gradfq_Aqz = 0.0
    fq_Aqz = 0.0
    Asz = 0.0
    gradfs_Asz = 0.0
    fs_Asz = 0.0

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            normFPR_x = blockvecnorm(sol.FPR_x)
            uppbnd = sol.f_Ax - blockvecdot(sol.At_gradf_Ax, sol.FPR_x) + 0.5/sol.gamma*normFPR_x^2;
            # TODO: we can save allocations in the next four lines
            Aqz = sol.Aq*sol.z
            gradfq_Aqz, fq_Aqz = gradient(sol.fq, Aqz)
            Asz = sol.As*sol.z
            gradfs_Asz, fs_Asz = gradient(sol.fs, Asz)
            f_Az = fs_Asz + fq_Aqz
            if f_Az > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                blockaxpy!(sol.y, sol.x, -sol.gamma, sol.At_gradf_Ax)
                sol.z, = prox(sol.g, sol.y, sol.gamma)
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
        extr = it/(it+3)
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

end

################################################################################
# Solver interface(s)

function FBS!(x0; kwargs...)
    sol = FBSIterator(x0; kwargs...)
    it = run(sol)
    blockcopy!(x0, sol.x)
    return (sol, it)
end

FBSSolver(; kwargs1...) = (x0; kwargs2...) -> FBS!(x0; kwargs1..., wargs2...)
