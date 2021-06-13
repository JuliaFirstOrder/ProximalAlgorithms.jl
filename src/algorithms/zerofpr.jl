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
- `H=LBFGS(x0, 5)`: variable metric to use to compute line-search directions.

# References
- [1] Themelis, Stella, Patrinos, "Forward-backward envelope for the sum of two
nonconvex functions: Further properties and nonmonotone line-search algorithms",
SIAM Journal on Optimization, vol. 28, no. 3, pp. 2274–2303 (2018).
"""

Base.@kwdef struct ZeroFPRIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,TA,Tg,TH}
    f::Tf = Zero()
    A::TA = I
    g::Tg = Zero()
    x0::Tx
    alpha::R = real(eltype(x0))(0.95)
    beta::R = real(eltype(x0))(0.5)
    Lf::Maybe{R} = nothing
    gamma::Maybe{R} = Lf === nothing ? nothing : (alpha / Lf)
    adaptive::Bool = false
    minimum_gamma::R = real(eltype(x0))(1e-7)
    max_backtracks::Int = 20
    H::TH = LBFGS(x0, 5)
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
    tau::Maybe{R} = nothing
    Axbar::TAx = zero(Ax)
    grad_f_Axbar::TAx = zero(Ax)
    At_grad_f_Axbar::Tx = zero(x)
    xbarbar::Tx = zero(x)
    res_xbar::Tx = zero(x)
    xbar_prev::Maybe{Tx} = nothing
    res_xbar_prev::Maybe{Tx} = nothing
    d::Tx = zero(x)
    Ad::TAx = zero(Ax)
end

f_model(state::ZeroFPRState) =
    f_model(state.f_Ax, state.At_grad_f_Ax, state.res, state.gamma)

function Base.iterate(iter::ZeroFPRIteration{R}) where {R}
    x = copy(iter.x0)
    Ax = iter.A * x
    grad_f_Ax, f_Ax = gradient(iter.f, Ax)

    gamma = iter.gamma

    if gamma === nothing
        # compute lower bound to Lipschitz constant of the gradient of x ↦ f(Ax)
        xeps = x .+ R(1)
        grad_f_Axeps, f_Axeps = gradient(iter.f, iter.A * xeps)
        L = norm(iter.A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
        gamma = iter.alpha / L
    end

    # compute initial forward-backward step
    At_grad_f_Ax = iter.A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    xbar, g_xbar = prox(iter.g, y, gamma)

    # compute initial fixed-point residual
    res = x - xbar

    state = ZeroFPRState(
        x=x, Ax=Ax, f_Ax=f_Ax, grad_f_Ax=grad_f_Ax, At_grad_f_Ax=At_grad_f_Ax,
        gamma=gamma, y=y, xbar=xbar, g_xbar=g_xbar, res=res, H=iter.H,
    )

    return state, state
end

function Base.iterate(
    iter::ZeroFPRIteration{R},
    state::ZeroFPRState{R,Tx,TAx},
) where {R,Tx,TAx}
    f_Axbar_upp = f_model(state)
    # These need to be performed anyway (to compute xbarbar later on)
    mul!(state.Axbar, iter.A, state.xbar)
    f_Axbar = gradient!(state.grad_f_Axbar, iter.f, state.Axbar)

    # backtrack gamma (warn and halt if gamma gets too small)
    while iter.gamma === nothing || iter.adaptive == true
        if state.gamma < iter.minimum_gamma
            @warn "parameter `gamma` became too small ($(state.gamma)), stopping the iterations"
            return nothing
        end
        tol = 10 * eps(R) * (1 + abs(f_Axbar))
        if f_Axbar <= f_Axbar_upp + tol
            break
        end
        state.gamma *= 0.5
        state.y .= state.x .- state.gamma .* state.At_grad_f_Ax
        state.g_xbar = prox!(state.xbar, iter.g, state.y, state.gamma)
        state.res .= state.x .- state.xbar
        reset!(state.H)
        f_Axbar_upp = f_model(state)
        mul!(state.Axbar, iter.A, state.xbar)
        f_Axbar = gradient!(state.grad_f_Axbar, iter.f, state.Axbar)
    end

    if state.xbar_prev === nothing
        state.xbar_prev = zero(state.x)
        state.res_xbar_prev = zero(state.x)
    end

    # compute FBE
    FBE_x = f_Axbar_upp + state.g_xbar

    # compute residual at xbar
    mul!(state.At_grad_f_Axbar, iter.A', state.grad_f_Axbar)
    state.y .= state.xbar .- state.gamma .* state.At_grad_f_Axbar
    g_xbarbar = prox!(state.xbarbar, iter.g, state.y, state.gamma)
    state.res_xbar .= state.xbar .- state.xbarbar

    if state.xbar_prev !== nothing
        # update metric
        update!(state.H, state.xbar - state.xbar_prev, state.res_xbar - state.res_xbar_prev)
        # store vectors for next update
        copyto!(state.xbar_prev, state.xbar)
        copyto!(state.res_xbar_prev, state.res_xbar)
    end

    # compute direction
    mul!(state.d, state.H, -state.res_xbar)

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
        FBE_x = f_model(state) + state.g_xbar

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
        (state.tau === nothing ? 0.0 : state.tau)
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
