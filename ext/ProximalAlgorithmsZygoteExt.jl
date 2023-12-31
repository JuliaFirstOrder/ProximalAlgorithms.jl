module ProximalAlgorithmsZygoteExt

using ProximalAlgorithms
using Zygote: pullback

struct ZygoteFunction{F}
    f::F
end

(f::ZygoteFunction)(x) = f.f(x)

function ProximalAlgorithms.eval_with_pullback(f::ZygoteFunction, x)
    out, pb = pullback(f, x)
    zygote_pullback() = pb(one(out))[1]
    return out, zygote_pullback
end

end
