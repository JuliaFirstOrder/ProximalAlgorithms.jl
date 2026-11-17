# Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for
# Convex Optimization", J. Optim. Theory Appl. (2018) (POGM, with the
# gradient-based adaptive restart of that paper's Section 5).

"""
    POGMIteration(; <keyword-arguments>)

Iterator implementing the proximal optimized gradient method (POGM) [1].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth. Currently only supports the `mf == 0`, fixed-stepsize case.

Unlike a proximal-gradient method, POGM's worst-case rate is *tight*: the
momentum it carries is the largest a first-order method can carry, so a
stepsize even slightly above `1/Lf` can make it diverge rather than merely
converge more slowly. Since an `Lf` obtained from a power iteration is an
under-estimate (the power method converges from below), `adaptive_restart`
is on by default. It resets the extrapolation parameter to 1 on either of the
two conditions of [1]: when `f + g` has risen from the previous iterate, and
when the generalized gradient at the new iterate points along the direction
just travelled — the momentum is fighting descent. Both are needed: with a
`Zero` `g` the prox is the identity, so the generalized gradient is identically
zero and only the function test can fire. Together they accelerate the typical
case and keep a slightly optimistic `Lf` from diverging.

See also: [`POGM`](@ref).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of `f`.
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).
- `adaptive_restart=true`: reset the extrapolation whenever the objective rises or the momentum stops pointing downhill, and back `gamma` off when it rises on consecutive iterations.
- `adaptive=(gamma === nothing)`: backtrack the stepsize against a descent condition every iteration, at the cost of one extra evaluation of `f`. On by default exactly when no stepsize was supplied, since the fallback `gamma` is then derived from a *lower* bound on the smoothness constant and is unsafe for POGM.
- `reduce_gamma=0.5`: factor the stepsize is multiplied by when that backoff fires.
- `minimum_gamma=1e-7`: floor for the stepsize backoff.

# References
1. Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for Convex Optimization", Journal of Optimization Theory and Applications (2018).
"""
Base.@kwdef struct POGMIteration{Tx,Tf,Tg,TLf,Tgamma}
    f::Tf = Zero()
    g::Tg = Zero()
    x0::Tx
    Lf::TLf = nothing
    gamma::Tgamma = Lf === nothing ? nothing : (1 / Lf)
    adaptive_restart::Bool = true
    adaptive::Bool = gamma === nothing
    reduce_gamma::Float64 = 0.5
    minimum_gamma::Float64 = 1.0e-7
end

Base.IteratorSize(::Type{<:POGMIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct POGMState{R,Tx}
    x::Tx             # iterate
    f_x::R            # value f at x
    grad_f_x::Tx      # gradient of f at x
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    z::Tx             # forward-backward point
    g_z::R            # value of g at z
    res::Tx           # fixed-point residual at iterate (= z - x)
    theta::R = one(gamma)  # extrapolation "theta" parameter
    y_prev::Tx = copy(y)   # forward point at the previous iteration
    w_prev::Tx = copy(x)   # pre-prox composite point at the previous iteration
    zeta_prev::R = gamma   # prox stepsize at the previous iteration
    F_prev::R = oftype(gamma, Inf)  # f + g at the previous iterate, for the function-based restart
    bad_streak::Int = 0    # consecutive function-restart triggers, for the stepsize backoff
end

function Base.iterate(iter::POGMIteration)
    x = copy(iter.x0)
    f_x, grad_f_x = value_and_gradient(iter.f, x)
    R = real(eltype(x))
    gamma = R(iter.gamma === nothing ? 1 / lower_bound_smoothness_constant(iter.f, I, x, grad_f_x) : iter.gamma)
    y = x - gamma .* grad_f_x
    # With no `Lf` to go on, `gamma` above comes from a *lower* bound on the smoothness constant,
    # i.e. it is an *upper* bound on the stepsize — the one direction POGM cannot survive. Back it
    # off to one that satisfies the descent condition before the momentum ever gets going.
    if iter.adaptive
        gamma, y, f_x = _pogm_backtrack(iter, x, f_x, grad_f_x, gamma, y)
    end

    # First POGM update: theta_0 = 1, so beta = 0 and only the "eta" (OGM)
    # momentum term is active; see [1] in the docstring above.
    theta = R(1)
    theta_new = (1 + sqrt(1 + 4 * theta^2)) / 2
    eta = theta / theta_new
    w = y .+ eta .* (y .- x)
    zeta = gamma * (1 + eta)
    z, g_z = prox(iter.g, w, zeta)

    state = POGMState(
        x = x,
        f_x = f_x,
        grad_f_x = grad_f_x,
        gamma = gamma,
        y = y,
        z = z,
        g_z = g_z,
        res = x - z,
        theta = theta_new,
        y_prev = copy(y),
        w_prev = w,
        zeta_prev = zeta,
    )
    return state, state
end

function Base.iterate(iter::POGMIteration, state::POGMState{R,Tx}) where {R,Tx}
    # Carry the previous prox output forward as the point where the
    # gradient is evaluated: POGM has no separate momentum-on-x step,
    # the momentum is folded directly into the pre-prox point below.
    state.x .= state.z
    state.f_x = value_and_gradient!(state.grad_f_x, iter.f, state.x)
    state.y .= state.x .- state.gamma .* state.grad_f_x
    if iter.adaptive
        state.gamma, state.y, state.f_x =
            _pogm_backtrack(iter, state.x, state.f_x, state.grad_f_x, state.gamma, state.y)
    end

    # Function-based restart ([1], Section 5). `state.g_z` is g at the point the previous prox
    # produced, which is exactly `state.x`, so f + g at the previous iterate costs nothing extra.
    # An increase means the extrapolation has overshot, and the momentum is dropped *before* this
    # step rather than after it. This is the condition that matters when g is `Zero`: the prox is
    # then the identity, so the gradient-based test below sees a zero generalized gradient and can
    # never fire, while a stepsize slightly above 1/Lf still diverges.
    F_x = state.f_x + state.g_z
    # `!(<=)` rather than `>`, so a `NaN` — which compares false against everything and would
    # otherwise sail past both tests — counts as a rise and triggers the backoff.
    if iter.adaptive_restart && !(F_x <= state.F_prev)
        state.theta = one(R)
        state.bad_streak += 1
        # A single rise is ordinary non-monotonicity and the restart alone handles it. Rises on
        # *consecutive* iterations mean the stepsize itself is too long — `Lf` was an
        # under-estimate — and no amount of restarting fixes that, so back `gamma` off. A run
        # with a valid stepsize restarts only occasionally and never twice in a row, so this
        # settles after a few reductions instead of shrinking without bound.
        if state.bad_streak >= 2 && state.gamma > iter.minimum_gamma
            state.gamma = max(state.gamma * iter.reduce_gamma, iter.minimum_gamma)
            state.y .= state.x .- state.gamma .* state.grad_f_x
            state.bad_streak = 0
        end
    elseif iter.adaptive_restart
        state.bad_streak = 0
    end
    state.F_prev = F_x

    theta_new = (1 + sqrt(1 + 4 * state.theta^2)) / 2
    beta = (state.theta - 1) / theta_new
    eta = state.theta / theta_new

    # Pre-prox composite point; safe to update w_prev in place since the
    # update is elementwise (see [1] in the docstring above).
    coef = beta * state.gamma / state.zeta_prev
    state.w_prev .=
        state.y .+ beta .* (state.y .- state.y_prev) .+ eta .* (state.y .- state.x) .-
        coef .* (state.x .- state.w_prev)
    zeta = state.gamma * (1 + beta + eta)

    state.y_prev .= state.y
    state.g_z = prox!(state.z, iter.g, state.w_prev, zeta)
    state.res .= state.x .- state.z
    state.zeta_prev = zeta
    # Gradient-based adaptive restart ([1], Section 5): `(w - z)/zeta` is the generalized
    # gradient at the new iterate, and `z - x` is the step just taken. A positive inner product
    # means the extrapolation is pushing against descent, and the extrapolation parameter is
    # reset to 1 — which zeroes `beta` on the next iteration, so both memory terms drop out.
    state.theta =
        (iter.adaptive_restart && _pogm_restart(state.w_prev, state.z, state.x)) ? one(R) :
        theta_new

    return state, state
end

# Halve `gamma` until the plain gradient step `y = x - gamma * grad` satisfies the descent
# condition `f(y) <= f(x) - (gamma/2) * ‖grad‖²`, which is what `gamma <= 1/L` guarantees locally.
# The check is on the gradient step alone, so it is independent of the momentum POGM adds after
# it, and it costs one extra evaluation of `f` per backtracking trial. Only reached when no `Lf`
# was supplied; with one, `gamma = 1/Lf` and nothing here runs.
function _pogm_backtrack(iter, x, f_x, grad_f_x, gamma, y)
    sq_grad = real(dot(grad_f_x, grad_f_x))
    f_y = iter.f(y)
    tol = 10 * eps(typeof(gamma)) * (1 + abs(f_x))
    while !(f_y <= f_x - (gamma / 2) * sq_grad + tol) && gamma > iter.minimum_gamma
        gamma = max(gamma * oftype(gamma, iter.reduce_gamma), oftype(gamma, iter.minimum_gamma))
        y = x .- gamma .* grad_f_x
        f_y = iter.f(y)
    end
    return gamma, y, f_x
end

# `real(dot(w - z, z - x)) > 0`, without materializing either difference.
function _pogm_restart(w, z, x)
    acc = zero(real(eltype(z)))
    @inbounds @simd for i in eachindex(z)
        acc += real(conj(w[i] - z[i]) * (z[i] - x[i]))
    end
    return acc > 0
end

default_stopping_criterion(tol, ::POGMIteration, state::POGMState) =
    norm(state.res, Inf) / state.gamma <= tol
default_solution(::POGMIteration, state::POGMState) = state.z
default_iteration_summary(it, ::POGMIteration, state::POGMState) =
    ("" => it, "f(x)" => state.f_x, "g(z)" => state.g_z, "‖x - z‖/γ" => norm(state.res, Inf) / state.gamma)

"""
    POGM(; <keyword-arguments>)

Constructs the proximal optimized gradient method (POGM) [1].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth. It has a worst-case rate a factor of 2 better than
FISTA's; see [`FastForwardBackward`](@ref) for the FISTA-type accelerated
forward-backward splitting algorithm, of which this is a momentum-update
variant. The gradient-based adaptive restart of [1] is on by default; see
[`POGMIteration`](@ref) for why.

The returned object has type `IterativeAlgorithm{POGMIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`POGMIteration`](@ref), [`FastForwardBackward`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-8`: tolerance for the default stopping criterion
- `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state
- `display::Function`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `POGMIteration` constructor upon call

# References
1. Kim, Fessler, "Adaptive Restart of the Optimized Gradient Method for Convex Optimization", Journal of Optimization Theory and Applications (2018).
"""
POGM(;
    maxit = nothing,
    tol = nothing,
    stop = nothing,
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    POGMIteration;
    termination_parameters(maxit, tol, stop; default_maxit = 10_000, default_tol = 1e-8)...,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

get_assumptions(::Type{<:POGMIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_convex)),
    SimpleTerm(:g => (is_proximable, is_convex,))
)
