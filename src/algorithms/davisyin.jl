################################################################################
# Davis-Yin splitting iterator
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

struct DY_iterable{R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}}
    f
    g
    h
    L
    x0::T
    gamma::R
    lambda::R
end

function DY_iterable()
    # TODO: auto set parameters here
end

mutable struct DY_state{T}
    x::T
    z::T
    temp_x::T
    res::T
end

function DY_state(iter::DY_iterable)
    # TODO: initialize state here
end

function Base.iterate(iter::DY_iterable, state::DY_state=DRS_state(iter))
    prox!(state.z, iter.g, state.x, iter.gamma)
    y .= iter.L * state.z
    grad_h_y, = gradient(iter.h, y)
    mul!(state.temp_x, state.L', grad_h_y)
    state.temp_x .*= -iter.gamma
    state.temp_x .+= 2 .* state.z .- state.x
    prox!(state.res, iter.f, state.temp_x, iter.gamma)
    state.res .-= state.z
    state.x .+= iter.lambda .* state.res
    return state, state
end
