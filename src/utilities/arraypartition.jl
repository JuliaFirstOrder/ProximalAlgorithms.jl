using ProximalOperators
using RecursiveArrayTools

## AbstractOperator

# import Base: *
# using LinearAlgebra
# using AbstractOperators
#
# @inline LinearAlgebra.mul!(y::ArrayPartition, L::AbstractOperator, x::ArrayPartition) =
#     LinearAlgebra.mul!(y.x, L, x.x)
#
# @inline (*)(L::AbstractOperator, x::ArrayPartition) = ArrayPartition(L * x.x)

## ProximableFunction

@inline function ProximalOperators.prox(h::ProximableFunction, x::ArrayPartition, gamma...)
    # unwrap
    y, fy = ProximalOperators.prox(h, x.x, gamma...)
    # wrap
    return ArrayPartition(y), fy
end

@inline function ProximalOperators.gradient(h::ProximableFunction, x::ArrayPartition)
    # unwrap
    grad, fx = ProximalOperators.gradient(h, x.x)
    # wrap
    return ArrayPartition(grad), fx
end

@inline ProximalOperators.prox!(y::ArrayPartition, h::ProximableFunction, x::ArrayPartition, gamma...) =
    ProximalOperators.prox!(y.x, h, x.x, gamma...)

@inline ProximalOperators.gradient!(y::ArrayPartition, h::ProximableFunction, x::ArrayPartition) =
    ProximalOperators.gradient!(y.x, h, x.x)
