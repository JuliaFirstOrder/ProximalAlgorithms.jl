using Zygote: pullback
using ProximalCore

struct ZygoteFunction{F}
    f::F
end

function ProximalCore.gradient!(grad, f::ZygoteFunction, x)
    fx, pb = pullback(f.f, x)
    grad .= pb(one(fx))[1]
    return fx
end
