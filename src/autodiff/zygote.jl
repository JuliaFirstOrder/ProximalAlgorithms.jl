using ProximalCore
using .Zygote: pullback

function ProximalCore.gradient!(grad_x, f::ZygoteFunction, x)
    f_x, pb = pullback(f.f, x)
    grad_x .= pb(one(f_x))[1]
    return f_x
end
