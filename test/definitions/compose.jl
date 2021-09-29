using ProximalOperators: ProximableFunction
using RecursiveArrayTools: ArrayPartition
import ProximalOperators: gradient!, gradient

struct ComposeAffine <: ProximableFunction
    f
    A
    b
end

function compose_affine_gradient!(y, g::ComposeAffine, x)
    res = g.A * x .+ g.b
    gradres, v = gradient(g.f, res)
    mul!(y, adjoint(g.A), gradres)
    return v
end

gradient!(y, g::ComposeAffine, x) = compose_affine_gradient!(y, g, x)
gradient!(y::ArrayPartition, g::ComposeAffine, x::ArrayPartition) = compose_affine_gradient!(y, g, x)

function ProximalOperators.gradient(h::ComposeAffine, x::ArrayPartition)
    grad_fx = similar(x)
    fx = gradient!(grad_fx, h, x)
    return grad_fx, fx
end
