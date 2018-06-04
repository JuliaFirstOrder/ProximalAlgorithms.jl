################################################################################
# Davis-Yin splitting iterator
#
# See:
# [1] Davis, Yin "A Three-Operator Splitting Scheme and its Optimization Applications",
# Set-Valued and Variational Analysis, vol. 25, no. 4, pp 829–858 (2017).
#

struct DavisYinIterator{I <: Integer, R <: Real, T1 <: BlockArray{R}, T2 <: BlockArray{R}} <: ProximalAlgorithm{I, Tuple{T1, T2}}
    z::T1
    f
    g
    h
    L
    gamma::R
    lambda::R
    maxit::I
    tol::R
    verbose::I
    verbose_freq::I
    x::T1
    y::T2
    z_half::T1
    w::T1
    FPR_x::T1
end

################################################################################
# Constructor(s)

function DavisYinIterator(z0::BlockArray{R}; f=Zero(), g=Zero(), h=Zero(), L=Identity(blocksize(z0)), gamma::R=1.0, maxit::I=10000, tol::R=1e-4, verbose=1, verbose_freq = 100) where {I, R}
    # TODO
end

################################################################################
# Utility methods

maxit(sol::DavisYinIterator) = sol.maxit

converged(sol::DavisYinIterator, it) = it > 0 && blockmaxabs(sol.FPR_x)/sol.gamma <= sol.tol

verbose(sol::DavisYinIterator) = sol.verbose > 0
verbose(sol::DavisYinIterator, it) = sol.verbose > 0 && (sol.verbose == 2 ? true : (it == 1 || it%sol.verbose_freq == 0))

function display(sol::DavisYinIterator)
    # TODO
end

function display(sol::DavisYinIterator, it)
    # TODO
end

function Base.show(io::IO, sol::DavisYinIterator)
    # TODO
end

################################################################################
# Initialization

function initialize!(sol::DavisYinIterator)
    return
end

################################################################################
# Iteration

function iterate!(sol::DavisYinIterator{I, T}, it::I) where {I, T}
    # x = prox(g, z)
    # y = Lx
    # w = ∇h(y)
    # z_half = 2x - z - γL'w
    # z = z + λ(prox(f, z_half) - x)
end

################################################################################
# Solver interface

function DavisYin(x0; kwargs...)
    sol = DavisYinIterator(x0; kwargs...)
    it, point = run!(sol)
    return (it, point, sol)
end
