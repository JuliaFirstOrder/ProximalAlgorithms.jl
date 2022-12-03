using ProximalCore
using .Yota: grad

function ProximalCore.gradient!(grad_x, f::YotaFunction, x)
    f_x, g = grad(f.f, x)
    grad_x .= g[2]
    return f_x
end
