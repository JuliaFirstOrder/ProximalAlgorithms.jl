using Zygote: pullback
using ProximalOperators

function ProximalOperators.gradient(f, x)
    fx, pb = pullback(f, x)
    grad = pb(one(fx))[1]
    return grad, fx
end

function ProximalOperators.gradient!(grad, f, x)
    y, fx = gradient(f, x)
    grad .= y
    return fx
end
