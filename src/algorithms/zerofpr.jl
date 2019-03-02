################################################################################
# ZeroFPR iterator (with L-BFGS directions)

mutable struct ZeroFPRIterator{I <: Integer, R <: Real, D, CS, FS, AS, CQ, FQ, AQ, G, HH} <: ProximalAlgorithm{I,D}
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
    verbose::I
    verbose_freq::I
    alpha::R
    beta::R
    tau::R
    y::D # gradient step
    xbar::D # proximal-gradient step
    xbarbar::D # proximal-gradient step
    xnew::D
    xnewbar::D
    H::HH # inverse Jacobian approximation
    FPR_x::D
    Aqx::CQ
    Asx::CS
    Aqxbar::CQ
    Asxbar::CS
    Aqxnew::CQ
    Asxnew::CS
    Aqd::CQ
    Asd::CS
    gradfq_Aqx::CQ
    gradfs_Asx::CS
    fs_Asx::R
    fq_Aqx::R
    f_Ax::R
    At_gradf_Ax::D
    Ast_gradfs_Asx::D
    Aqt_gradfq_Aqx::D
    g_xbar::R
    FBE_x::R
    FPR_xbar::D
    FPR_xbar_prev::D
    d::D
end

################################################################################
# Constructor

function ZeroFPRIterator(x0::D; 
                         fs::FS=Zero(), As::AS=Identity(size(x0)), 
                         fq::FQ=Zero(), Aq::AQ=Identity(size(x0)), 
                         g::G=Zero(), 
                         gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, adaptive::Bool=false, memory::I=10, 
                         verbose::I=1, verbose_freq::I=100, alpha::R=0.95, beta::R=0.5) where {I, R, D, FS, AS, FQ, AQ, G}
    x = copy(x0)
    y = zero(x0)
    xbar = zero(x0)
    xbarbar = zero(x0)
    xnew = zero(x0)
    xnewbar = zero(x0)
    xbar = zero(x0)
    FPR_x = zero(x0)
    FPR_xbar_prev = zero(x0)
    FPR_xbar = zero(x0)
    Aqx = Aq*x
    Asx = As*x
    Aqxbar = zero(Aqx)
    Asxbar = zero(Asx)
    Aqxnew = zero(Aqx)
    Asxnew = zero(Asx)
    Aqd = zero(Aqx)
    Asd = zero(Asx)
    gradfq_Aqx = zero(Aqx)
    gradfs_Asx = zero(Asx)
    At_gradf_Ax = zero(x0)
    Ast_gradfs_Asx = zero(x0)
    Aqt_gradfq_Aqx = zero(x0)
    d = zero(x0)
    H = LBFGS(x, memory)
    CQ = typeof(Aqx)
    CS = typeof(Asx)
    HH = typeof(H)
    ZeroFPRIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}(x,
                 fs, As,
                 fq, Aq, g,
                 gamma, maxit, tol, adaptive, verbose, verbose_freq,
                 alpha, beta, one(R), y,
                 xbar, xbarbar, xnew, xnewbar,
                 H, FPR_x,
                 Aqx, Asx,
                 Aqxbar, Asxbar,
                 Aqxnew, Asxnew,
                 Aqd, Asd,
                 gradfq_Aqx, gradfs_Asx,
                 zero(R), zero(R), zero(R),
                 At_gradf_Ax, Ast_gradfs_Asx, Aqt_gradfq_Aqx,
                 zero(R), zero(R),
                 FPR_xbar, FPR_xbar_prev, d)
end

################################################################################
# Utility methods

maxit(sol::ZeroFPRIterator{I}) where {I} = sol.maxit

converged(sol::ZeroFPRIterator{I,R,D}, it::I) where {I,R,D} = it > 0 && maximum(abs,sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::ZeroFPRIterator) = sol.verbose > 0
verbose(sol::ZeroFPRIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::ZeroFPRIterator)
    @printf("%6s | %10s | %10s | %10s | %10s |\n ", "it", "gamma", "fpr", "tau", "FBE")
    @printf("------|------------|------------|------------|------------|\n")
end
function display(sol::ZeroFPRIterator, it)
    @printf("%6d | %7.4e | %7.4e | %7.4e | %7.4e | \n", it, sol.gamma, maximum(abs,sol.FPR_x)/sol.gamma, sol.tau, sol.FBE_x)
end

function Base.show(io::IO, sol::ZeroFPRIterator)
    println(io, "ZeroFPR" )
    println(io, "fpr        : $(maximum(abs,sol.FPR_x))")
    println(io, "gamma      : $(sol.gamma)")
    println(io, "tau        : $(sol.tau)")
    print(  io, "FBE        : $(sol.FBE_x)")
end

################################################################################
# Initialization

function initialize!(sol::ZeroFPRIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}) where {I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}

    # reset L-BFGS operator (would be nice to have this option)
    # TODO add function reset!(::LBFGS) in AbstractOperators
    sol.H.currmem, sol.H.curridx = 0, 0
    sol.H.H = 1.0

    # compute first forward-backward step here
    mul!(sol.Aqx, sol.Aq, sol.x)
    sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqx)
    mul!(sol.Asx, sol.As, sol.x)
    sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asx)
    sol.At_gradf_Ax .= sol.As'*sol.gradfs_Asx .+ sol.Aq'*sol.gradfq_Aqx
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
        sol.gamma = sol.alpha/L
    end

    sol.y .= sol.x .- sol.gamma .* sol.At_gradf_Ax
    sol.g_xbar = prox!(sol.xbar, sol.g, sol.y, sol.gamma)
    sol.FPR_x .= sol.x .- sol.xbar

    return sol.xbar

end

################################################################################
# Iteration

function iterate!(sol::ZeroFPRIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}, it::I) where {I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}

    # These need to be performed anyway (to compute xbarbar later on)
    mul!(sol.Aqxbar, sol.Aq, sol.xbar)
    fq_Aqxbar = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxbar)
    mul!(sol.Asxbar, sol.As, sol.xbar)
    fs_Asxbar = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxbar)
    f_Axbar = fs_Asxbar + fq_Aqxbar

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            normFPR_x = norm(sol.FPR_x)
            uppbnd = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_x)) + 0.5/sol.gamma*normFPR_x^2
            if f_Axbar > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                sol.y .= sol.x .- sol.gamma .* sol.At_gradf_Ax
                sol.xbar, sol.g_xbar = prox(sol.g, sol.y, sol.gamma)
                sol.FPR_x .= sol.x .- sol.xbar
            else
                break
            end
            mul!(sol.Aqxbar, sol.Aq, sol.xbar)
            fq_Aqxbar = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxbar)
            mul!(sol.Asxbar, sol.As, sol.xbar)
            fs_Asxbar = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxbar)
            f_Axbar = fs_Asxbar + fq_Aqxbar
        end
    end

    # Compute value of FBE at x

    normFPR_x = norm(sol.FPR_x)
    FBE_x = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_x)) + 0.5/sol.gamma*normFPR_x^2 + sol.g_xbar

    # Compute search direction
    mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)
    mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
    sol.At_gradf_Ax .= sol.Ast_gradfs_Asx .+ sol.Aqt_gradfq_Aqx
    sol.y .= sol.xbar .- sol.gamma .* sol.At_gradf_Ax
    g_xbarbar = prox!(sol.xbarbar, sol.g, sol.y, sol.gamma)
    sol.FPR_xbar .= sol.xbar .- sol.xbarbar

    if it > 1
        update!(sol.H, sol.xbar, sol.xnewbar, sol.FPR_xbar, sol.FPR_xbar_prev)
    end
    mul!(sol.d, sol.H, ( x -> .-x ).(sol.FPR_xbar)) # TODO: not nice

    # Perform line-search over the FBE

    sol.tau = 1.0

    mul!(sol.Asd, sol.As, sol.d)
    mul!(sol.Aqd, sol.Aq, sol.d)

    sigma = 0.5*sol.beta/sol.gamma*(1.0-sol.alpha)

    g_xnewbar = zero(R)
    f_Axnew = zero(R)
    FBE_xnew = zero(R)

    maxit_tau = 10
    for it_tau = 1:maxit_tau # TODO: replace/complement with lower bound on tau
        sol.xnew .= sol.xbar .+ sol.tau .* sol.d
        sol.Asxnew .= sol.Asxbar .+ sol.tau .* sol.Asd
        sol.Aqxnew .= sol.Aqxbar .+ sol.tau .* sol.Aqd
        fs_Asxnew = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxnew)
        # TODO: can precompute most of next line before the iteration
        fq_Aqxnew = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxnew)
        f_Axnew = fs_Asxnew + fq_Aqxnew
        mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)
        mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
        sol.At_gradf_Ax .= sol.Ast_gradfs_Asx .+ sol.Aqt_gradfq_Aqx
        sol.y .= sol.xnew  .- sol.gamma .* sol.At_gradf_Ax
        g_xnewbar = prox!(sol.xnewbar, sol.g, sol.y, sol.gamma)
        sol.FPR_x .= sol.xnew .- sol.xnewbar
        normFPR_xnew = norm(sol.FPR_x)
        FBE_xnew = f_Axnew - real(dot(sol.At_gradf_Ax, sol.FPR_x)) + 0.5/sol.gamma*normFPR_xnew^2 + g_xnewbar
        if FBE_xnew <= FBE_x - sigma*normFPR_x^2
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
