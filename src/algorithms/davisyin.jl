################################################################################
# Davis-Yin splitting iterable
#
# See:
# [1] Davis, Yin "A Three-Operator Splitting Scheme and its Optimization Applications",
# Set-Valued and Variational Analysis, vol. 25, no. 4, pp 829–858 (2017).
#

using Base.Iterators
using ProximalAlgorithms.IterationTools
using ProximalOperators: Zero
using LinearAlgebra
using Printf

struct DYS_iterable{R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}, Tf, Tg, Th, TA}
    f::Tf
    g::Tg
    h::Th
    A::TA
    x0::T
    gamma::R
    lambda::R
end

mutable struct DYS_state{T, S}
    z::T
    xg::T
    y::S
    grad_h_y::S
    z_half::T
    xf::T
    res::T
end

function Base.iterate(iter::DYS_iterable{R, C, T}) where {R, C, T}
    z = copy(iter.x0)
    xg, = prox(iter.g, z, iter.gamma)
    y = iter.A * xg
    grad_h_y, = gradient(iter.h, y)
    z_half = 2 .* xg .- z .- iter.gamma .* (iter.A' * grad_h_y)

    # TODO: check the following
    xf = similar(z_half)
    prox!(xf, iter.f, z_half, iter.gamma)
    # xf, = prox(iter.f, z_half, iter.gamma)  # this fails the tests
    ############################################################################

    res = xf - xg
    z .+= iter.lambda .* res
    state = DYS_state{T, typeof(y)}(z, xg, y, grad_h_y, z_half, xf, res)
    return state, state
end

function Base.iterate(iter::DYS_iterable, state::DYS_state)
    prox!(state.xg, iter.g, state.z, iter.gamma)
    mul!(state.y, iter.A, state.xg)
    gradient!(state.grad_h_y, iter.h, state.y)
    mul!(state.z_half, iter.A', state.grad_h_y)
    state.z_half .*= -iter.gamma
    state.z_half .+= 2 .* state.xg .- state.z
    prox!(state.xf, iter.f, state.z_half, iter.gamma)
    state.res .= state.xf .- state.xg
    state.z .+= iter.lambda .* state.res
    return state, state
end

"""
    davisyin(x0; f, g, h, A, [...])

Solves convex optimization problems of the form

    minimize f(x) + g(x) + h(A x),

where `h` is smooth and `A` is a linear mapping, using the Davis-Yin splitting
algorithm, see [1].

Either of the following arguments must be specified:

* `L::Real`, Lipschitz constant of the gradient of `h(A x)`.
* `gamma:Real`, stepsize parameter.

Other optional keyword arguments:

* `labmda::Real` (default: `1.0`), relaxation parameter, see [1].
* `maxit::Integer` (default: `1000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

References:

[1] Davis, Yin. "A Three-Operator Splitting Scheme and its Optimization
Applications", Set-Valued and Variational Analysis, vol. 25, no. 4,
pp. 829–858 (2017).
"""
function davisyin(x0;
    f=Zero(), g=Zero(), h=Zero(), A=I,
    lambda=1.0, L=nothing, gamma=nothing,
    maxit=10_000, tol=1e-8,
    verbose=false, freq=100)

    R = real(eltype(x0))

    stop(state::DYS_state) = norm(state.res, Inf) <= R(tol)
    disp((it, state)) = @printf("%5d | %.3e\n", it, norm(state.res, Inf))

    if gamma === nothing
        if L !== nothing
            gamma = R(1)/R(L)
        else
            error("You must specify either L or gamma")
        end
    end

    iter = DYS_iterable(f, g, h, A, x0, R(gamma), R(lambda))
    iter = take(halt(iter, stop), maxit)
    iter = enumerate(iter)
    if verbose iter = tee(sample(iter, freq), disp) end

    num_iters, state_final = loop(iter)

    return state_final.xf, state_final.xg, num_iters
end
