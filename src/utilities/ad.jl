using Zygote: pullback
using ProximalCore

function ProximalCore.gradient!(grad, f, x)
    fx, pb = pullback(f, x)
    grad .= pb(one(fx))[1]
    return fx
end
