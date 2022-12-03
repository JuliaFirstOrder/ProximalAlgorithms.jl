using ProximalCore
using .Yota: grad

struct YotaFunction{F}
    f::F
end

(f::YotaFunction)(x) = f.f(x)

function ProximalCore.gradient!(grad_x, f::YotaFunction, x)
    f_x, g = grad(f.f, x)
    grad_x .= g[2]
    return f_x
end
