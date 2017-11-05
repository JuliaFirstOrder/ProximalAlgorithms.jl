################################################################################
# Template iterator

struct TemplateIterator{I <: Integer, T <: BlockArray} <: ProximalAlgorithm{I, T}
    x::T
    maxit::I
    # Put here problem description, other parameters, method memory,
    # also maybe storage for in-place computations, etc...
end

################################################################################
# Constructor(s)

TemplateIterator(x0::T; maxit::I=10) where {I, T} = TemplateIterator{I, T}(x0, maxit)

################################################################################
# Utility methods

maxit(sol::TemplateIterator) = sol.maxit

converged(sol::TemplateIterator, it) = false

verbose(sol::TemplateIterator, it) = true

display(sol::TemplateIterator, it) = println("$(it) iterations performed")

################################################################################
# Initialization

function initialize(sol::TemplateIterator)
    # One shouldn't really be printing anything here
    println("Initializing the iterations")
end

################################################################################
# Iteration

function iterate(sol::TemplateIterator{I, T}, it::I) where {I, T}
    # One shouldn't really be printing anything here
    println("Performing one iteration")
    return sol.x
end

################################################################################
# Solver interface(s)

function Template(x0; kwargs...)
    # Create iterable
    sol = TemplateIterator(x0; kwargs...)
    # Run iterations
    (it, point) = run(sol)
    return (it, point, sol)
end

# The following is for *partially* defining the solver
TemplateSolver(; kwargs1...) = (x0; kwargs2...) -> Template(x0; kwargs1..., kwargs2...)
