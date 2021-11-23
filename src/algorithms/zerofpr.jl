# Themelis, Stella, Patrinos, "Forward-backward envelope for the sum of two
# nonconvex functions: Further properties and nonmonotone line-search
# algorithms", SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274–2303
# (2018).

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

"""
    ZeroFPRIteration(; <keyword-arguments>)

Instantiate the ZeroFPR algorithm (see [1]) for solving optimization problems
of the form

    minimize f(Ax) + g(x),

where `f` is smooth and `A` is a linear mapping (for example, a matrix).

# Arguments
- `x0`: initial point.
- `f=Zero()`: smooth objective term.
- `A=I`: linear operator (e.g. a matrix).
- `g=Zero()`: proximable objective term.
- `Lf=nothing`: Lipschitz constant of the gradient of x ↦ f(Ax).
- `gamma=nothing`: stepsize to use, defaults to `1/Lf` if not set (but `Lf` is).
- `adaptive=false`: forces the method stepsize to be adaptively adjusted.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `max_backtracks=20`: maximum number of line-search backtracks.
- `directions=LBFGS(5)`: strategy to use to compute line-search directions.

# References
- [1] Themelis, Stella, Patrinos, "Forward-backward envelope for the sum of two
nonconvex functions: Further properties and nonmonotone line-search algorithms",
SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274–2303 (2018).
"""

Base.@kwdef struct ZeroFPRIteration{R,Tx,Tf,TA,Tg,D}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (alpha / Lf)
    adaptive::Bool = gamma === nothing
    minimum_gamma::R = real(eltype(x0))(1e-7)
    max_backtracks::Int = 20
    directions::D = LBFGS(5)
end

Base.IteratorSize(::Type{<:ZeroFPRIteration}) = Base.IsInfinite()

Base.@kwdef mutable struct ZeroFPRState{R,Tx,TAx,TH}
    x::Tx             # iterate
    Ax::TAx           # A times x
    f_Ax::R           # value of smooth term
    grad_f_Ax::TAx    # gradient of f at Ax
    At_grad_f_Ax::Tx  # gradient of smooth term
    gamma::R          # stepsize parameter of forward and backward steps
    y::Tx             # forward point
    xbar::Tx          # forward-backward point
    g_xbar::R         # value of nonsmooth term (at xbar)
    res::Tx           # fixed-point residual at iterate (= x - xbar)
    H::TH             # variable metric
    tau::R = zero(gamma)
    Axbar::TAx = similar(Ax)
    grad_f_Axbar::TAx = similar(Ax)
    At_grad_f_Axbar::Tx = similar(x)
    xbarbar::Tx = similar(x)
    res_xbar::Tx = similar(x)
    xbar_prev::Tx = similar(x)
    res_xbar_prev::Tx = similar(x)
    is_prev_set::Bool = false
    d::Tx = similar(x)
    Ad::TAx = similar(Ax)
end

f_model(iter::ZeroFPRIteration, state::ZeroFPRState) = f_model(state.f_Ax, state.At_grad_f_Ax, state.res, iter.alpha / state.gamma)

function Base.iterate(iter::ZeroFPRIteration{R}) where R
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)
    gamma = iter.gamma === nothing ? iter.alpha / lower_bound_smoothness_constant(iter.f, iter.A, x, grad_f_Ax) : iter.gamma
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    xbar, g_xbar = prox(iter.g, y, gamma)
    state = ZeroFPRState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, xbar=xbar, g_xbar=g_xbar, res=x - xbar, H=initialize(iter.directions, x),
    )
    return state, state
end

set_next_direction!(::QuasiNewtonStyle, ::ZeroFPRIteration, state::ZeroFPRState) = mul!(state.d, state.H, -state.res_xbar)
set_next_direction!(::NoAccelerationStyle, ::ZeroFPRIteration, state::ZeroFPRState) = state.d .= .-state.res
set_next_direction!(iter::ZeroFPRIteration, state::ZeroFPRState) = set_next_direction!(acceleration_style(typeof(iter.directions)), iter, state)

update_direction_state!(::QuasiNewtonStyle, ::ZeroFPRIteration, state::ZeroFPRState) = update!(state.H, state.xbar - state.xbar_prev, state.res_xbar - state.res_xbar_prev)
update_direction_state!(::NoAccelerationStyle, ::ZeroFPRIteration, state::ZeroFPRState) = return
update_direction_state!(iter::ZeroFPRIteration, state::ZeroFPRState) = update_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

reset_direction_state!(::QuasiNewtonStyle, ::ZeroFPRIteration, state::ZeroFPRState) = reset!(state.H)
reset_direction_state!(::NoAccelerationStyle, ::ZeroFPRIteration, state::ZeroFPRState) = return
reset_direction_state!(iter::ZeroFPRIteration, state::ZeroFPRState) = reset_direction_state!(acceleration_style(typeof(iter.directions)), iter, state)

function Base.iterate(iter::ZeroFPRIteration{R}, state::ZeroFPRState) where R
    f_Axbar_upp, f_Axbar = if iter.adaptive == true
        gamma_prev = state.gamma
        state.gamma, state.g_xbar, f_Axbar, f_Axbar_upp = backtrack_stepsize!(
            state.gamma, iter.f, iter.A, iter.g,
            state.x, state.f_Ax, state.At_grad_f_Ax, state.y, state.xbar, state.g_xbar, state.res,
            state.Axbar, state.grad_f_Axbar,
            alpha = iter.alpha, minimum_gamma = iter.minimum_gamma,
        )
        if state.gamma != gamma_prev
            reset_direction_state!(iter, state)
        end
        f_Axbar_upp, f_Axbar
    else
        mul!(state.Axbar, iter.A, state.xbar)
        f_model(iter, state), gradient!(state.grad_f_Axbar, iter.f, state.Axbar)
    end

    # compute FBE
    FBE_x = f_Axbar_upp + state.g_xbar

    # compute residual at xbar
    mul!(state.At_grad_f_Axbar, iter.A', state.grad_f_Axbar)
    state.y .= state.xbar .- state.gamma .* state.At_grad_f_Axbar
    g_xbarbar = prox!(state.xbarbar, iter.g, state.y, state.gamma)
    state.res_xbar .= state.xbar .- state.xbarbar

    if state.is_prev_set == true
        update_direction_state!(iter, state)
    end

    copyto!(state.xbar_prev, state.xbar)
    copyto!(state.res_xbar_prev, state.res_xbar)
    state.is_prev_set = true

    set_next_direction!(iter, state)

    # Perform line-search over the FBE
    tau = R(1)
    mul!(state.Ad, iter.A, state.d)

    sigma = iter.beta * (0.5 / state.gamma) * (1 - iter.alpha)
    tol = 10 * eps(R) * (1 + abs(FBE_x))
    threshold = FBE_x - sigma * norm(state.res)^2 + tol

    for _ in 1:iter.max_backtracks
        state.x .= state.xbar_prev .+ tau .* state.d
        state.Ax .= state.Axbar .+ tau .* state.Ad
        # TODO: can precompute most of next line in case f is quadratic
        state.f_Ax = gradient!(state.grad_f_Ax, iter.f, state.Ax)
        mul!(state.At_grad_f_Ax, iter.A', state.grad_f_Ax)
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_xbar = prox!(state.xbar, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.xbar
        FBE_x = f_model(iter, state) + state.g_xbar

        if FBE_x <= threshold
            state.tau = tau
            return state, state
        end

        tau *= 0.5
    end

    @warn "stepsize `tau` became too small ($(tau)), stopping the iterations"
    return nothing
end

# Solver

struct ZeroFPR{R, K}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    kwargs::K
end

function (solver::ZeroFPR)(x0; kwargs...)
    stop(state::ZeroFPRState) = norm(state.res, Inf) / state.gamma <= solver.tol
    disp((it, state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it,
        state.gamma,
        norm(state.res, Inf) / state.gamma,
        state.tau,
    )
    iter = ZeroFPRIteration(; x0=x0, solver.kwargs..., kwargs...)
    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter, solver.freq), disp)
    end
    num_iters, state_final = loop(iter)
    return state_final.xbar, num_iters
end

ZeroFPR(; maxit=1_000, tol=1e-8, verbose=false, freq=10, kwargs...) = 
    ZeroFPR(maxit, tol, verbose, freq, kwargs)
