import ProximalOperators
import RecursiveArrayTools

@inline function ProximalOperators.prox(
    h::ProximalOperators.ProximableFunction,
    x::RecursiveArrayTools.ArrayPartition,
    gamma...
)
    # unwrap
    y, fy = ProximalOperators.prox(h, x.x, gamma...)
    # wrap
    return RecursiveArrayTools.ArrayPartition(y), fy
end

@inline function ProximalOperators.gradient(
    h::ProximalOperators.ProximableFunction,
    x::RecursiveArrayTools.ArrayPartition
)
    # unwrap
    grad, fx = ProximalOperators.gradient(h, x.x)
    # wrap
    return RecursiveArrayTools.ArrayPartition(grad), fx
end

@inline ProximalOperators.prox!(
    y::RecursiveArrayTools.ArrayPartition,
    h::ProximalOperators.ProximableFunction,
    x::RecursiveArrayTools.ArrayPartition,
    gamma...
) = ProximalOperators.prox!(y.x, h, x.x, gamma...)

@inline ProximalOperators.gradient!(
    y::RecursiveArrayTools.ArrayPartition,
    h::ProximalOperators.ProximableFunction,
    x::RecursiveArrayTools.ArrayPartition
) = ProximalOperators.gradient!(y.x, h, x.x)
