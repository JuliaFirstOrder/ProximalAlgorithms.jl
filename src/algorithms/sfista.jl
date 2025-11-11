# An implementation of a FISTA-like method, where the smooth part of the objective function can be strongly convex.

"""
    SFISTAIteration(; <keyword-arguments>)

Iterator implementing the FISTA-like algorithm in [3].

This iterator solves strongly convex composite optimization problems of the form

    minimize f(x) + g(x),

where g is proper closed convex and f is a continuously differentiable function that is `mf`-strongly convex and whose gradient is
`Lf`-Lipschitz continuous.

The scheme is based on Nesterov's accelerated gradient method [1, Eq. (4.9)] and Beck's method for the convex case [2]. Its full
definition is given in [3, Algorithm 2.2.2.], and some analyses of this method are given in [3, 4, 5]. Another perspective is that
it is a special instance of [4, Algorithm 1] in which μh=0.

See also: [`SFISTA`](@ref).

# Arguments
- `x0`: initial point; must be in the domain of g.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `mf=0` : strong convexity constant of f (see above).
- `Lf` : Lipschitz constant of ∇f (see above).

# References
1. Nesterov, Y. (2013). Gradient methods for minimizing composite functions. Mathematical Programming, 140(1), 125-161.
2. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
3. Kong, W. (2021). Accelerated Inexact First-Order Methods for Solving Nonconvex Composite Optimization Problems. arXiv preprint arXiv:2104.09685.
4. Kong, W., Melo, J. G., & Monteiro, R. D. (2021). FISTA and Extensions - Review and New Insights. arXiv preprint arXiv:2107.01267.
5. Florea, M. I. (2018). Constructing Accelerated Algorithms for Large-scale Optimization-Framework, Algorithms, and Applications.
"""
Base.@kwdef struct SFISTAIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Th}
    x0::Tx
    f::Tf = Zero()
    g::Th = Zero()
    Lf::R
    mf::R = real(eltype(Lf))(0.0)
    termination_type::Symbol = :classic # can be :AIPP or :classic (default)
end

Base.IteratorSize(::Type{<:SFISTAIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct SFISTAState{R,Tx}
    λ::R                                # stepsize.
    yPrev::Tx                           # previous main iterate.
    y::Tx = zero(yPrev)                # main iterate.
    xPrev::Tx = copy(yPrev)             # previous auxiliary iterate.
    x::Tx = zero(yPrev)                # auxiliary iterate (see [3]).
    xt::Tx = zero(yPrev)                # prox center used to generate main iterate y.
    τ::R = real(eltype(yPrev))(1.0)     # helper variable (see [3]).
    a::R = real(eltype(yPrev))(0.0)     # helper variable (see [3]).
    APrev::R = real(eltype(yPrev))(1.0) # previous A (helper variable).
    A::R = real(eltype(yPrev))(0.0)     # helper variable (see [3]).
    gradf_xt::Tx = zero(yPrev)          # array containing ∇f(xt).
    res_norm::R = real(eltype(yPrev))(0.0) # norm of the residual (for stopping criterion).
end

function Base.iterate(
    iter::SFISTAIteration,
    state::SFISTAState = SFISTAState(λ = 1 / iter.Lf, yPrev = copy(iter.x0)),
)
    # Set up helper variables.
    state.τ = state.λ * (1 + iter.mf * state.APrev)
    state.a = (state.τ + sqrt(state.τ^2 + 4 * state.τ * state.APrev)) / 2
    state.A = state.APrev + state.a
    state.xt .= (state.APrev / state.A) .* state.yPrev + (state.a / state.A) .* state.xPrev
    f_xt, gradf_xt = value_and_gradient(iter.f, state.xt)
    state.gradf_xt .= gradf_xt
    λ2 = state.λ / (1 + state.λ * iter.mf)
    # FISTA acceleration steps.
    prox!(state.y, iter.g, state.xt - λ2 * state.gradf_xt, λ2)
    state.x .=
        state.xPrev .+
        (state.a / (1 + state.A * iter.mf)) .*
        ((state.y .- state.xt) ./ state.λ .+ iter.mf .* (state.y .- state.xPrev))
    # Update state variables.
    state.yPrev .= state.y
    state.xPrev .= state.x
    state.APrev = state.A
    return state, state
end

# Different stopping conditions (sc). Returns the current residual value and whether or not a stopping condition holds.
function calc_residual!(state::SFISTAState, iter::SFISTAIteration)
    if iter.termination_type == :AIPP
        # AIPP-style termination [4]. The main inclusion is: r ∈ ∂_η(f + h)(y).
        r = (iter.y0 - state.x) / state.A
        η = (norm(iter.y0 - state.y)^2 - norm(state.x - state.y)^2) / (2 * state.A)
        res = (norm(r)^2 + max(η, 0.0)) / max(norm(iter.y0 - state.y + r)^2, 1e-16)
    else
        # Classic (approximate) first-order stationary point [4]. The main inclusion is: r ∈ ∇f(y) + ∂h(y).
        λ2 = state.λ / (1 + state.λ * iter.mf)
        f_y, gradf_y = value_and_gradient(iter.f, state.y)
        r = gradf_y - state.gradf_xt + (state.xt - state.y) / λ2
        res = norm(r)
    end
    state.res_norm = res
end

default_stopping_criterion(tol, iter::SFISTAIteration, state::SFISTAState) = begin
    calc_residual!(state, iter)
    state.res_norm <= tol || state.res_norm ≈ tol
end
default_solution(::SFISTAIteration, state::SFISTAState) = state.y
default_iteration_summary(it, iter::SFISTAIteration, state::SFISTAState) =
    ("" => it, (iter.termination_type == :AIPP ? "‖∂_η(f + h)(y)‖" : "‖∇f(y) + ∂h(y)‖") => state.res_norm)

"""
    SFISTA(; <keyword-arguments>)

Constructs the the FISTA-like algorithm in [3].

This algorithm solves strongly convex composite optimization problems of the form

    minimize f(x) + g(x),

where g is proper closed convex and f is a continuously differentiable function that is `mf`-strongly convex and whose gradient is
Lf-Lipschitz continuous.

The scheme is based on Nesterov's accelerated gradient method [1, Eq. (4.9)] and Beck's method for the convex case [2]. Its full
definition is given in [3, Algorithm 2.2.2.], and some analyses of this method are given in [3, 4, 5]. Another perspective is that
it is a special instance of [4, Algorithm 1] in which μh=0.

The returned object has type `IterativeAlgorithm{SFISTAIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`SFISTAIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-6`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `SFISTAIteration` constructor upon call

# References
1. Nesterov, Y. (2013). Gradient methods for minimizing composite functions. Mathematical Programming, 140(1), 125-161.
2. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.
3. Kong, W. (2021). Accelerated Inexact First-Order Methods for Solving Nonconvex Composite Optimization Problems. arXiv preprint arXiv:2104.09685.
4. Kong, W., Melo, J. G., & Monteiro, R. D. (2021). FISTA and Extensions - Review and New Insights. arXiv preprint arXiv:2107.01267.
5. Florea, M. I. (2018). Constructing Accelerated Algorithms for Large-scale Optimization-Framework, Algorithms, and Applications.
"""
SFISTA(;
    maxit = 10_000,
    tol = 1e-6,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    SFISTAIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:SFISTAIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_strongly_convex)),
    SimpleTerm(:g => (is_proximable, is_convex)),
)
