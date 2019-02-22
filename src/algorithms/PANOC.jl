################################################################################
# PANOC iterator (with L-BFGS directions)

mutable struct PANOCIterator{I <: Integer, R <: Real, D, CS, FS, AS, CQ, FQ, AQ, G, HH} <: ProximalAlgorithm{I,D}
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
    H::HH # inverse Jacobian approximation
    FPR_x::D
    Aqx::CQ
    Asx::CS
    Aqxnew::CQ
    Asxnew::CS
    Aqd::CQ
    Asd::CS
    Aqfb::CQ
    Asfb::CS
    gradfq_Aqx::CQ
    gradfs_Asx::CS
    Aqxbar::CQ
    Asxbar::CS
    gradfq_Aqxbar::CQ
    gradfs_Asxbar::CS
    fs_Asx::R
    fq_Aqx::R
    f_Ax::R
    At_gradf_Ax::D
    Aqt_gradfq_Aqx::D
    Ast_gradfs_Asx::D
    g_xbar::R
    FBE_x::R
    FBE_x_prev::R
    x_prev::D
    FPR_x_prev::D
    xnew::D
    xnewbar::D
    FPR_xnew::D
    d::D
    normFPR_x::R
end

################################################################################
# Constructor

function PANOCIterator(x0::D; 
                       fs::FS=Zero(), As::AS=Identity(size(x0)), 
                       fq::FQ=Zero(), Aq::AQ=Identity(size(x0)), 
                       g::G=Zero(), 
                       gamma::R=-1.0, maxit::I=10000, tol::R=1e-4, adaptive::Bool=false, 
                       memory::I=10, verbose::I=1, verbose_freq::I=100, 
                       alpha::R=0.95, beta::R=0.5) where {I, R, D, FS, AS, FQ, AQ, G}
    x       = copy(x0)
    xbar    = zero(x0)
    x_prev  = zero(x0)
    xnew    = zero(x0)
    xnewbar = zero(x0)
    y = zero(x0)
    FPR_x = zero(x0)
    FPR_x_prev = zero(x0)
    FPR_xnew = zero(x0)
    Aqx = Aq*x
    Asx = As*x
    Aqxnew = zero(Aqx)
    Asxnew = zero(Asx)
    Aqd = zero(Aqx)
    Asd = zero(Asx)
    Aqfb = zero(Aqx)
    Asfb = zero(Asx)
    gradfq_Aqx = zero(Aqx)
    gradfs_Asx = zero(Asx)
    Aqxbar = zero(Aqx)
    Asxbar = zero(Asx)
    gradfq_Aqxbar = zero(Aqx)
    gradfs_Asxbar = zero(Asx)
    At_gradf_Ax = zero(x0)
    Aqt_gradfq_Aqx = zero(x0)
    Ast_gradfs_Asx = zero(x0)
    d = zero(x0)
    H = LBFGS(x, memory)
    CQ = typeof(Aqx)
    CS = typeof(Asx)
    HH = typeof(H)
    PANOCIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}(
               x, fs, As,
               fq, Aq, g,
               gamma, maxit, tol,
               adaptive, verbose,
               verbose_freq, alpha, beta,
               one(R), y, xbar,
               H, FPR_x,
               Aqx, Asx, Aqxnew, Asxnew, Aqd, Asd, Aqfb, Asfb, gradfq_Aqx, gradfs_Asx,
               Aqxbar, Asxbar, gradfq_Aqxbar, gradfs_Asxbar,
               zero(R), zero(R), zero(R),
               At_gradf_Ax, Aqt_gradfq_Aqx, Ast_gradfs_Asx,
               zero(R), zero(R),
               zero(R), x_prev, FPR_x_prev,
               xnew, xnewbar, FPR_xnew,
               d, zero(R))
end

################################################################################
# Utility methods

maxit(sol::PANOCIterator{I}) where {I} = sol.maxit

converged(sol::PANOCIterator{I,R,D}, it::I)  where {I,R,D}= it > 0 && maximum(abs,sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::PANOCIterator) = sol.verbose > 0
verbose(sol::PANOCIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::PANOCIterator)
    @printf("%6s | %10s | %10s | %10s | %10s |\n ", "it", "gamma", "fpr", "tau", "FBE")
    @printf("------|------------|------------|------------|------------|\n")
end

function display(sol::PANOCIterator, it)
    @printf("%6d | %7.4e | %7.4e | %7.4e | %7.4e | \n", it, sol.gamma, maximum(abs,sol.FPR_x)/sol.gamma, sol.tau, sol.FBE_x)
end

function Base.show(io::IO, sol::PANOCIterator)
    println(io, "PANOC" )
    println(io, "fpr        : $(maximum(abs,sol.FPR_x))")
    println(io, "gamma      : $(sol.gamma)")
    println(io, "tau        : $(sol.tau)")
    print(  io, "FBE        : $(sol.FBE_x)")
end

################################################################################
# Initialization

function initialize!(sol::PANOCIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}) where {I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}

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

    sol.y .= sol.x .-sol.gamma .* sol.At_gradf_Ax
    sol.g_xbar = prox!(sol.xbar, sol.g, sol.y, sol.gamma)
    sol.FPR_x .= sol.x .- sol.xbar

    sol.normFPR_x = norm(sol.FPR_x)
    sol.FBE_x = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_x)) + 0.5/sol.gamma*sol.normFPR_x^2 + sol.g_xbar

    return sol.xbar

end

################################################################################
# Iteration

function iterate!(sol::PANOCIterator{I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}, it::I) where {I, R, D, CS, FS, AS, CQ, FQ, AQ, G, HH}

    if sol.adaptive
        for it_gam = 1:100 # TODO: replace/complement with lower bound on gamma
            mul!(sol.Aqxbar, sol.Aq, sol.xbar)
            fq_Aqxbar = gradient!(sol.gradfq_Aqxbar, sol.fq, sol.Aqxbar)
            mul!(sol.Asxbar, sol.As, sol.xbar)
            fs_Asxbar = gradient!(sol.gradfs_Asxbar, sol.fs, sol.Asxbar)
            f_Axbar = fs_Asxbar + fq_Aqxbar

            uppbnd = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_x)) +
                 0.5/sol.gamma*sol.normFPR_x^2
            if f_Axbar > uppbnd + 1e-6*abs(sol.f_Ax)
                sol.gamma = 0.5*sol.gamma
                sol.y .= sol.x .- sol.gamma .* sol.At_gradf_Ax
                sol.g_xbar = prox!(sol.xbar, sol.g, sol.y, sol.gamma)
                sol.FPR_x .= sol.x .- sol.xbar
                sol.normFPR_x = norm(sol.FPR_x)
            else
                sol.FBE_x = uppbnd + sol.g_xbar
                break
            end
        end
    end

    if it > 1
        update!(sol.H, sol.x, sol.x_prev, sol.FPR_x, sol.FPR_x_prev)
    end
    mul!(sol.d, sol.H, ( x -> .-x ).(sol.FPR_x)) # TODO: not nice

    sol.FPR_x_prev, sol.FPR_x = sol.FPR_x, sol.FPR_x_prev
    sol.x_prev .= sol.x

    sigma = 0.5*sol.beta/sol.gamma*(1.0-sol.alpha)
    maxit_tau = 10

    # tau = 1
    sol.tau = one(R)

    mul!( sol.Aqd, sol.Aq, sol.d)
    mul!( sol.Asd, sol.As, sol.d)

    # xnew = x + tau*d
    sol.xnew .= sol.x .+ sol.tau .* sol.d
    # Aq*xnew = Aq*x + tau*Aq*d
    sol.Aqxnew .= sol.Aqx .+ sol.tau .* sol.Aqd
    # As*xnew = As*x + tau*As*d
    sol.Asxnew .= sol.Asx .+ sol.tau .* sol.Asd

    # calculate new FBE in xnew
    sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxnew)
    sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxnew)

    mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
    mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)

    sol.At_gradf_Ax .= sol.Aqt_gradfq_Aqx .+ sol.Ast_gradfs_Asx
    sol.f_Ax = sol.fs_Asx + sol.fq_Aqx

    # gradient step
    sol.y .= sol.xnew .- sol.gamma .* sol.At_gradf_Ax
    # prox step
    sol.g_xbar = prox!(sol.xnewbar, sol.g, sol.y, sol.gamma)

    sol.FPR_xnew .= sol.xnew .- sol.xnewbar
    norm_FPRxnew = norm(sol.FPR_xnew)

    FBE_xnew = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_xnew)) +
                   0.5/sol.gamma*norm_FPRxnew^2 + sol.g_xbar

    if FBE_xnew > sol.FBE_x - sigma*sol.normFPR_x^2
        # start using convex combination of FB direction and d

        mul!(sol.Aqfb, sol.Aq, sol.FPR_x_prev)
        mul!(sol.Asfb, sol.As, sol.FPR_x_prev)

        for it_tau = 1:maxit_tau # TODO: replace/complement with lower bound on tau

            sol.tau *= 0.5

            # xnew = x + tau*d
            sol.xnew .= sol.x .+ sol.tau .* sol.d
            # xnew = x + tau*d - (1-tau)*fb
            sol.xnew .= sol.xnew .+ (sol.tau-1.0) .* sol.FPR_x_prev

            # Aq*xnew = Aq*x + tau*Aq*d
            sol.Aqxnew .= sol.Aqx .+ sol.tau .* sol.Aqd
            # Aq*xnew = Aq*x + tau*Aq*d - (1-tau)*Aq*fb
            sol.Aqxnew .= sol.Aqxnew .+ (sol.tau-1.0) .* sol.Aqfb

            # As*xnew = As*x + tau*As*d
            sol.Asxnew .= sol.Asx .+ sol.tau .* sol.Asd
            # As*xnew = As*x + tau*As*d - (1-tau)*As*fb
            sol.Asxnew .= sol.Asxnew .+ (sol.tau-1.0) .* sol.Asfb

            # calculate new FBE in xnew
            sol.fq_Aqx = gradient!(sol.gradfq_Aqx, sol.fq, sol.Aqxnew)
            sol.fs_Asx = gradient!(sol.gradfs_Asx, sol.fs, sol.Asxnew)

            mul!(sol.Aqt_gradfq_Aqx, sol.Aq', sol.gradfq_Aqx)
            mul!(sol.Ast_gradfs_Asx, sol.As', sol.gradfs_Asx)

            sol.At_gradf_Ax .= sol.Aqt_gradfq_Aqx .+ sol.Ast_gradfs_Asx
            sol.f_Ax = sol.fs_Asx + sol.fq_Aqx

            # gradient step
            sol.y .= sol.xnew .- sol.gamma .* sol.At_gradf_Ax
            # prox step
            sol.g_xbar = prox!(sol.xnewbar, sol.g, sol.y, sol.gamma)

            sol.FPR_xnew .= sol.xnew .- sol.xnewbar
            norm_FPRxnew = norm(sol.FPR_xnew)

            FBE_xnew = sol.f_Ax - real(dot(sol.At_gradf_Ax, sol.FPR_xnew)) +
                   0.5/sol.gamma*norm_FPRxnew^2 + sol.g_xbar

            if FBE_xnew <= sol.FBE_x - sigma*sol.normFPR_x^2
                break
            end

        end

    end

    sol.normFPR_x = norm_FPRxnew
    sol.FPR_x, sol.FPR_xnew = sol.FPR_xnew, sol.FPR_x
    sol.FBE_x = FBE_xnew
    sol.x, sol.xbar, sol.xnew, sol.xnewbar = sol.xnew, sol.xnewbar, sol.x, sol.xbar
    sol.Aqx, sol.Aqxnew = sol.Aqxnew, sol.Aqx
    sol.Asx, sol.Asxnew = sol.Asxnew, sol.Asx

    return sol.xbar

end

################################################################################
# Solver interface(s)

function PANOC(x0; kwargs...)
    sol = PANOCIterator(x0; kwargs...)
    it, point = run!(sol)
    x0 .= point
    return (it, point, sol)
end
