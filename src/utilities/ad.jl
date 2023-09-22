using Zygote: pullback
using ProximalCore

struct ZygoteFunction{F}
    f::F
end

(f::ZygoteFunction)(x) = f.f(x)

function ProximalCore.gradient!(grad, f::ZygoteFunction, x)
    fx, pb = pullback(f.f, x)
    grad .= pb(one(fx))[1]
    return fx
end
