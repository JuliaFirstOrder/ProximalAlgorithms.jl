# Latafat, Patrinos, "Asymmetric forward–backward–adjoint splitting for
# solving monotone inclusions involving three operators", Computational
# Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017).
#
# Latafat, Patrinos, "Primal-dual proximal algorithms for structured convex
# optimization: a unifying framework", In Large-Scale and Distributed 
# Optimization, Giselsson and Rantzer, Eds. Springer International Publishing,
# pp. 97–120 (2018).
#
# Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
# with Applications to Imaging", Journal of Mathematical Imaging and Vision,
# vol. 40, no. 1, pp. 120-145 (2011).
#
# Condat, "A primal–dual splitting method for convex optimization
# involving Lipschitzian, proximable and linear composite terms",
# Journal of Optimization Theory and Applications, vol. 158, no. 2,
# pp 460-479 (2013).
# 
# Vũ, "A splitting algorithm for dual monotone inclusions involving
# cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3,
# pp. 667-681 (2013).

"""
    AFBAIteration(; <keyword-arguments>)

Iterator implementing the asymmetric forward-backward-adjoint algorithm (AFBA, see [1]).

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly
convex. Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.

Points `x0` and `y0` are the initial primal and dual iterates, respectively.
If unspecified, functions `f`, `g`, and `h` default to the identically zero
function, `l` defaults to the indicator of the set `{0}`, and `L` defaults to
the identity. Important keyword arguments, in case `f` and `l` are set, are
the Lipschitz constants `beta_f` and `beta_l` (see below).

The iterator implements Algorithm 3 of [1] with constant stepsize (α_n=λ)
for several prominant special cases:
1) θ = 2          ==>   Corresponds to the Vu-Condat Algorithm [3, 4].
2) θ = 1, μ=1
3) θ = 0, μ=1
4) θ ∈ [0,∞), μ=0

See [2, Section 5.2] and [1, Figure 1] for stepsize conditions, special cases,
and relation to other algorithms.

See also: [`AFBA`](@ref).

# Arguments
- `x0`: initial primal point.
- `y0`: initial dual point.
- `f=Zero()`: smooth objective term.
- `g=Zero()`: proximable objective term.
- `h=Zero()`: proximable objective term.
- `l=IndZero()`: strongly convex function.
- `L=I`: linear operator (e.g. a matrix).
- `beta_f=0`: Lipschitz constant of the gradient of `f`.
- `beta_l=0`: Lipschitz constant of the gradient of `l` conjugate.
- `theta=1`: nonnegative algorithm parameter.
- `mu=1`: algorithm parameter in the range [0,1].
- `gamma1`: primal stepsize (see [1] for the default choice).
- `gamma2`: dual stepsize (see [1] for the default choice).

# References
1. Latafat, Patrinos, "Asymmetric forward-backward-adjoint splitting for solving monotone inclusions involving three operators", Computational Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017).
2. Latafat, Patrinos, "Primal-dual proximal algorithms for structured convex optimization: a unifying framework", In Large-Scale and Distributed Optimization, Giselsson and Rantzer, Eds. Springer International Publishing, pp. 97-120 (2018).
3. Condat, "A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms", Journal of Optimization Theory and Applications, vol. 158, no. 2, pp 460-479 (2013).
4. Vũ, "A splitting algorithm for dual monotone inclusions involving cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3, pp. 667-681 (2013).
"""
Base.@kwdef struct AFBAIteration{R,Tx,Ty,Tf,Tg,Th,Tl,TL,Tbetaf,Tbetal,Ttheta,Tmu,Tlambda}
    f::Tf = Zero()
    g::Tg = Zero()
    h::Th = Zero()
    l::Tl = IndZero()
    L::TL = if isa(h, Zero)
        L = 0 * I
    else
        I
    end
    x0::Tx
    y0::Ty = L * x0
    beta_f::Tbetaf = if isa(f, Zero)
        real(eltype(x0))(0)
    else
        error("argument beta_f must be specified together with f")
    end
    beta_l::Tbetal = if isa(l, IndZero)
        real(eltype(x0))(0)
    else
        error("argument beta_l must be specified together with l")
    end
    theta::Ttheta = real(eltype(x0))(1)
    mu::Tmu = real(eltype(x0))(1)
    lambda::Tlambda = real(eltype(x0))(1)
    gamma::Tuple{R,R} = if lambda != 1
        error("if lambda != 1, then you need to provide stepsizes manually")
    else
        T = real(eltype(x0))
        AFBA_default_stepsizes(L, h, T(theta), T(mu), T(beta_f), T(beta_l))
    end
end

Base.IteratorSize(::Type{<:AFBAIteration}) = Base.IsInfinite()

get_assumptions(::Type{<:AFBAIteration}) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_convex)),
    SimpleTerm(:g => (is_proximable, is_convex)),
    OperatorTermWithInfimalConvolution(:h => (is_proximable, is_convex), :l => (is_proximable, is_strongly_convex), :L => (is_linear,))
)

"""
    VuCondatIteration(; <keyword-arguments>)

Iterator implementing the Vũ-Condat primal-dual algorithm [1, 2].

This iterator solves convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly
convex. Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.

This iteration is equivalent to [`AFBAIteration`](@ref) with `theta=2`;
for all other arguments see [`AFBAIteration`](@ref).

See also: [`AFBAIteration`](@ref), [`VuCondat`](@ref).

# References
1. Condat, "A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms", Journal of Optimization Theory and Applications, vol. 158, no. 2, pp 460-479 (2013).
2. Vũ, "A splitting algorithm for dual monotone inclusions involving cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3, pp. 667-681 (2013).
"""
VuCondatIteration(; kwargs...) = AFBAIteration(kwargs..., theta = 2)

get_assumptions(::typeof(VuCondatIteration)) = AssumptionGroup(
    SimpleTerm(:f => (is_smooth, is_convex)),
    SimpleTerm(:g => (is_proximable, is_convex)),
    OperatorTermWithInfimalConvolution(:h => (is_proximable, is_convex), :l => (is_proximable, is_strongly_convex), :L => (is_linear,))
)

"""
    ChambollePockIteration(; <keyword-arguments>)

Iterator implementing the Chambolle-Pock primal-dual algorithm [1].

This iterator solves convex optimization problems of the form

    minimize g(x) + h(L x),

where `g` and `h` are possibly nonsmooth, and `L` is a linear mapping.

See also: [`AFBAIteration`](@ref), [`ChambollePock`](@ref).

This iteration is equivalent to [`AFBAIteration`](@ref) with `theta=2`, `f=Zero()`, `l=IndZero()`;
for all other arguments see [`AFBAIteration`](@ref).

# References
1. Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011).
"""
ChambollePockIteration(; kwargs...) =
    AFBAIteration(kwargs..., theta = 2, f = Zero(), l = IndZero())

get_assumptions(::T) where {T<:typeof(ChambollePockIteration)} = AssumptionGroup(
    SimpleTerm(:g => (is_proximable, is_convex)),
    OperatorTerm(:h => (is_proximable, is_convex), :L => (is_linear,))
)

Base.@kwdef struct AFBAState{Tx,Ty}
    x::Tx
    y::Ty
    xbar::Tx = similar(x)
    ybar::Ty = similar(y)
    gradf::Tx = similar(x)
    gradl::Ty = similar(y)
    FPR_x::Tx = similar(x)
    FPR_y::Ty = similar(y)
    temp_x::Tx = similar(x)
    temp_y::Ty = similar(y)
end

function Base.iterate(
    iter::AFBAIteration,
    state::AFBAState = AFBAState(x = copy(iter.x0), y = copy(iter.y0)),
)
    # perform xbar-update step
    f_x, gradf = value_and_gradient(iter.f, state.x)
    state.gradf .= gradf
    mul!(state.temp_x, iter.L', state.y)
    state.temp_x .+= state.gradf
    state.temp_x .*= -iter.gamma[1]
    state.temp_x .+= state.x
    prox!(state.xbar, iter.g, state.temp_x, iter.gamma[1])

    # perform ybar-update step
    lc_y, gradl = value_and_gradient(convex_conjugate(iter.l), state.y)
    state.gradl .= gradl
    state.temp_x .= iter.theta .* state.xbar .+ (1 - iter.theta) .* state.x
    mul!(state.temp_y, iter.L, state.temp_x)
    state.temp_y .-= state.gradl
    state.temp_y .*= iter.gamma[2]
    state.temp_y .+= state.y
    prox!(state.ybar, convex_conjugate(iter.h), state.temp_y, iter.gamma[2])

    # the residues
    state.FPR_x .= state.xbar .- state.x
    state.FPR_y .= state.ybar .- state.y

    # perform x-update step
    state.temp_y .= (iter.mu * (2 - iter.theta) * iter.gamma[1]) .* state.FPR_y
    mul!(state.temp_x, iter.L', state.temp_y)
    state.x .+= iter.lambda .* (state.FPR_x .- state.temp_x)

    # perform y-update step
    state.temp_x .= ((1 - iter.mu) * (2 - iter.theta) * iter.gamma[2]) .* state.FPR_x
    mul!(state.temp_y, iter.L, state.temp_x)
    state.y .+= iter.lambda .* (state.FPR_y .+ state.temp_y)

    return state, state
end

default_stopping_criterion(tol, ::AFBAIteration, state::AFBAState) =
    norm(state.FPR_x, Inf) + norm(state.FPR_y, Inf) <= tol
default_solution(::AFBAIteration, state::AFBAState) = (state.xbar, state.ybar)
default_iteration_summary(it, ::AFBAIteration, state::AFBAState) =
    ("" => it, "‖x̄ - x‖" => norm(state.FPR_x, Inf), "‖ȳ - y‖" => norm(state.FPR_y, Inf))

"""
    AFBA(; <keyword-arguments>)

Constructs the asymmetric forward-backward-adjoint algorithm (AFBA, see [1]).

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly
convex. Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{AFBAIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`AFBAIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-5`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AFBAIteration` constructor upon call

# References
1. Latafat, Patrinos, "Asymmetric forward-backward-adjoint splitting for solving monotone inclusions involving three operators", Computational Optimization and Applications, vol. 68, no. 1, pp. 57-93 (2017).
2. Latafat, Patrinos, "Primal-dual proximal algorithms for structured convex optimization: a unifying framework", In Large-Scale and Distributed Optimization, Giselsson and Rantzer, Eds. Springer International Publishing, pp. 97-120 (2018).
"""
AFBA(;
    maxit = 10_000,
    tol = 1e-5,
    stop = (iter, state) -> default_stopping_criterion(tol, iter, state),
    solution = default_solution,
    verbose = false,
    freq = 100,
    summary = default_iteration_summary,
    display = default_display,
    kwargs...,
) = IterativeAlgorithm(
    AFBAIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    summary,
    display,
    kwargs...,
)

"""
    VuCondat(; <keyword-arguments>)

Constructs the Vũ-Condat primal-dual algorithm [1, 2].

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x) + (h □ l)(L x),

where `f` is smooth, `g` and `h` are possibly nonsmooth and `l` is strongly
convex. Symbol `□` denotes the infimal convolution, and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{AFBAIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`VuCondatIteration`](@ref), [`AFBAIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-5`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AFBAIteration` constructor upon call

# References
1. Condat, "A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms", Journal of Optimization Theory and Applications, vol. 158, no. 2, pp 460-479 (2013).
2. Vũ, "A splitting algorithm for dual monotone inclusions involving cocoercive operators", Advances in Computational Mathematics, vol. 38, no. 3, pp. 667-681 (2013).
"""
VuCondat(; kwargs...) = AFBA(; kwargs..., theta = 2)

"""
    ChambollePock(; <keyword-arguments>)

Constructs the Chambolle-Pock primal-dual algorithm [1].

This algorithm solves convex optimization problems of the form

    minimize g(x) + h(L x),

where `g` and `h` are possibly nonsmooth, and `L` is a linear mapping.

The returned object has type `IterativeAlgorithm{AFBAIteration}`,
and can be called with the problem's arguments to trigger its solution.

See also: [`ChambollePockIteration`](@ref), [`AFBAIteration`](@ref), [`IterativeAlgorithm`](@ref).

# Arguments
- `maxit::Int=10_000`: maximum number of iteration
- `tol::1e-5`: tolerance for the default stopping criterion
- `stop::Function=(iter, state) -> default_stopping_criterion(tol, iter, state)`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
- `solution::Function=default_solution`: solution mapping, `solution(::T, state)` should return the identified solution
- `verbose::Bool=false`: whether the algorithm state should be displayed
- `freq::Int=100`: every how many iterations to display the algorithm state. If `freq <= 0`, only the final iteration is displayed.
- `summary::Function=default_iteration_summary`: function to generate iteration summaries, `summary(::Int, iter::T, state)` should return a summary of the iteration state
- `display::Function=default_display`: display function, `display(::Int, ::T, state)` should display a summary of the iteration state
- `kwargs...`: additional keyword arguments to pass on to the `AFBAIteration` constructor upon call

# References
1. Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", Journal of Mathematical Imaging and Vision, vol. 40, no. 1, pp. 120-145 (2011).
"""
ChambollePock(; kwargs...) = AFBA(; kwargs..., f = Zero(), l = IndZero(), theta = 2)

function AFBA_default_stepsizes(L, h::Zero, theta::R, mu::R, beta_f::R, beta_l::R) where {R}
    return R(1.99) / beta_f, R(1)
end

function AFBA_default_stepsizes(L, h, theta::R, mu::R, beta_f::R, beta_l::R) where {R}
    par = R(5) # scaling parameter for comparing Lipschitz constants and \|L\|
    par2 = R(100)   # scaling parameter for α
    alpha = R(1)
    nmL = R(opnorm(L))

    if theta ≈ 2 # default stepsize for Vu-Condat
        if nmL > par * max(beta_l, beta_f)
            alpha = R(1)
        elseif beta_f > par * beta_l
            alpha = par2 * nmL / beta_f
        elseif beta_l > par * beta_f
            alpha = beta_l / (par2 * nmL)
        end
        gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
        gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
    elseif theta ≈ 1 && mu ≈ 1 # SPCA
        if nmL > par2 * beta_l # for the case beta_f = 0
            alpha = R(1)
        elseif beta_l > par * beta_f
            alpha = beta_l / (par2 * nmL)
        end
        gamma1 = beta_f > 0 ? R(1.99) / beta_f : R(1) / (nmL / alpha)
        gamma2 = R(0.99) / (beta_l / 2 + gamma1 * nmL^2)
    elseif theta ≈ 0 && mu ≈ 1 # PPCA
        temp = R(3)
        if beta_f ≈ 0
            nmL *= sqrt(temp)
            if nmL > par * beta_l
                alpha = R(1)
            else
                alpha = beta_l / (par2 * nmL)
            end
            gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
        else
            if nmL > par * max(beta_l, beta_f)
                alpha = R(1)
            elseif beta_f > par * beta_l
                alpha = par2 * nmL / beta_f
            elseif beta_l > par * beta_f
                alpha = beta_l / (par2 * nmL)
            end
            xi = 1 + 2 * nmL / (nmL + alpha * beta_f / 2)
            gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + xi * nmL * alpha)
        end
    elseif mu ≈ 0 # SDCA & PDCA
        temp = theta^2 - 3 * theta + 3
        if beta_l ≈ 0
            nmL *= sqrt(temp)
            if nmL > par * beta_f
                alpha = R(1)
            else
                alpha = par2 * nmL / beta_f
            end
            gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
        else
            if nmL > par * max(beta_l, beta_f)
                alpha = R(1)
            elseif beta_f > par * beta_l
                alpha = par2 * nmL / beta_f
            elseif beta_l > par * beta_f
                alpha = beta_l / (par2 * nmL)
            end
            eta = 1 + (temp - 1) * alpha * nmL / (alpha * nmL + beta_l / 2)
            gamma1 = R(1) / (beta_f / 2 + eta * nmL / alpha)
            gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
        end
    elseif theta ≈ 0 && mu ≈ 0.5 # PPDCA
        if beta_l ≈ 0 || beta_f ≈ 0
            if nmL > par * max(beta_l, beta_f)
                alpha = R(1)
            elseif beta_f > par * beta_l
                alpha = par2 * nmL / beta_f
            elseif beta_l > par * beta_f
                alpha = beta_l / (par2 * nmL)
            end
        else
            alpha = sqrt(beta_l / beta_f) / 2
        end
        gamma1 = R(1) / (beta_f / 2 + nmL / alpha)
        gamma2 = R(0.99) / (beta_l / 2 + nmL * alpha)
    else
        error("this choice of theta and mu is not supported!")
    end

    return gamma1, gamma2
end
