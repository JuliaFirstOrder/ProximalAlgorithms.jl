import ProximalCore
import RecursiveArrayTools

@inline function ProximalCore.prox(h, x::RecursiveArrayTools.ArrayPartition, gamma...)
    # unwrap
    y, fy = ProximalCore.prox(h, x.x, gamma...)
    # wrap
    return RecursiveArrayTools.ArrayPartition(y), fy
end

@inline function ProximalCore.gradient(h, x::RecursiveArrayTools.ArrayPartition)
    # unwrap
    grad, fx = ProximalCore.gradient(h, x.x)
    # wrap
    return RecursiveArrayTools.ArrayPartition(grad), fx
end

@inline ProximalCore.prox!(y::RecursiveArrayTools.ArrayPartition, h, x::RecursiveArrayTools.ArrayPartition, gamma...) = ProximalCore.prox!(y.x, h, x.x, gamma...)

@inline ProximalCore.gradient!(y::RecursiveArrayTools.ArrayPartition, h, x::RecursiveArrayTools.ArrayPartition) = ProximalCore.gradient!(y.x, h, x.x)
