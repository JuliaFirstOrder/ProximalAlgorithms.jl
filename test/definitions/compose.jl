using RecursiveArrayTools: ArrayPartition
using ProximalCore

struct ComposeAffine
    f
    A
    b
end

function (g::ComposeAffine)(x)
    res = g.A * x .+ g.b
    return g.f(res)
end

function compose_affine_gradient!(y, g::ComposeAffine, x)
    res = g.A * x .+ g.b
    gradres, v = ProximalCore.gradient(g.f, res)
    mul!(y, adjoint(g.A), gradres)
    return v
end

ProximalCore.gradient!(y, g::ComposeAffine, x) = compose_affine_gradient!(y, g, x)
ProximalCore.gradient!(y::ArrayPartition, g::ComposeAffine, x::ArrayPartition) = compose_affine_gradient!(y, g, x)

function ProximalCore.gradient(h::ComposeAffine, x::ArrayPartition)
    grad_fx = similar(x)
    fx = ProximalCore.gradient!(grad_fx, h, x)
    return grad_fx, fx
end
