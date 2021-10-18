using Zygote: pullback
import ProximalOperators: gradient, gradient!

function gradient(f, x)
    fx, pb = pullback(f, x)
    grad = pb(one(fx))[1]
    return grad, fx
end

function gradient!(grad, f, x)
    y, fx = gradient(f, x)
    grad .= y
    return fx
end
