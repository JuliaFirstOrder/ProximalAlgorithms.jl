################################################################################
# Douglas-Rachford splitting iterator
#
# See:
# [1] Eckstein, Bertsekas "On the Douglas-Rachford Splitting Method and the Proximal Point Algorithm for Maximal Monotone Operators*", Mathematical Programming, vol. 55, no. 1, pp. 293-318 (1989).
#

struct DRSIterator{I <: Integer, R <: Real, T <: BlockArray{R}} <: ProximalAlgorithm{I, T}
    x::T
    f
    g
    gamma::R
    maxit::I
    tol::R
    verbose::I
    y::T
    r::T
    z::T
    FPR_x::T
end

################################################################################
# Constructor(s)

function DRSIterator(x0::BlockArray{R}; f=Zero(), g=Zero(), gamma::R=1.0, maxit::I=10000, tol::R=1e-4, verbose=2) where {I, R}
    y = blockcopy(x0)
    r = blockcopy(x0)
    z = blockcopy(x0)
    FPR_x = blockcopy(x0)
    FPR_x .= Inf
    return DRSIterator{I, R, typeof(x0)}(x0, f, g, gamma, maxit, tol, verbose, y, r, z, FPR_x)
end

################################################################################
# Utility methods

maxit(sol::DRSIterator) = sol.maxit

converged(sol::DRSIterator, it) = blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::DRSIterator)     = sol.verbose > 0
verbose(sol::DRSIterator, it) = sol.verbose > 0 && (sol.verbose == 1 ? true : (it == 1 || it%100 == 0))

function display(sol::DRSIterator)
	@printf("%6s | %10s | %10s |\n ", "it", "gamma", "fpr")
	@printf("------|------------|------------|\n")
end

function display(sol::DRSIterator, it)
    @printf("%6d | %7.4e | %7.4e | \n", it, sol.gamma, blockmaxabs(sol.FPR_x)/sol.gamma)
end

function Base.show(io::IO, sol::DRSIterator)
	println(io, "Douglas-Rachford Splitting" )
	println(io, "fpr        : $(blockmaxabs(sol.FPR_x))")
	println(io, "gamma      : $(sol.gamma)")
end

################################################################################
# Initialization

function initialize(sol::DRSIterator)
    return
end

################################################################################
# Iteration

function iterate(sol::DRSIterator{I, T}, it::I) where {I, T}
    prox!(sol.y, sol.f, sol.x, sol.gamma)
    sol.r .= 2.*sol.y .- sol.x
    prox!(sol.z, sol.g, sol.r, sol.gamma)
    sol.FPR_x .= sol.y .- sol.z
    sol.x .-= sol.FPR_x
    return sol.z
end

################################################################################
# Solver interface

function DRS(x0; kwargs...)
    sol = DRSIterator(x0; kwargs...)
    (it, point) = run(sol)
    return (it, point, sol)
end
