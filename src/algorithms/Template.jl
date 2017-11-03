################################################################################
# Template iterator

struct TemplateIterator <: ProximalAlgorithm
    x
    maxit::Integer
    # Put here problem description, other parameters, method memory,
    # also maybe storage for in-place computations, etc...
end

################################################################################
# Constructor(s)

TemplateIterator(x0; maxit=10) = TemplateIterator(x0, maxit)

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

function iterate(sol::TemplateIterator, it)
    # One shouldn't really be printing anything here
    println("Performing one iteration")
end

################################################################################
# Solver interface(s)

function Template!(x0; kwargs...)
    # Create iterable
    sol = TemplateIterator(x0; kwargs...)
    # Run iterations
    it = run(sol)
    return (sol, it)
end

# The following is for *partially* defining the solver
TemplateSolver(; kwargs1...) = (x0; kwargs2...) -> Template!(x0; kwargs1..., kwargs2...)
