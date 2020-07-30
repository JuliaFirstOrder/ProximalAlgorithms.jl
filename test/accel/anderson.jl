using Test

@testset "Anderson accel. ($R)" for R in [Float32, Float64]
    using LinearAlgebra
    using ProximalAlgorithms: AndersonAcceleration, update!

    n, k = 20, 10

    Q = randn(R, n, k)
    H = Q * Q' + I
    l = randn(R, n)

    f(x) = dot(x, H * x) / 2 + dot(x, l)

    x_star = -H \ l
    f_star = f(x_star)
    norm_x_star = norm(x_star)

    Lip = opnorm(H)
    gamma = 1 / Lip
    x = zeros(n)
    res = zeros(n)
    x_prev = zeros(n)
    res_prev = zeros(n)

    acc = AndersonAcceleration(x, 5)

    mul!(res, H, x)
    res .+= l
    res .*= gamma

    d = acc * res

    for it = 1:30
        # store iterate and residual for the operator update later
        res_prev .= res
        x_prev .= x

        # compute accelerated direction
        mul!(d, acc, res)

        # update iterate
        x .-= d

        # compute new residual
        mul!(res, H, x)
        res .+= l
        res .*= gamma

        # update operator
        update!(acc, x - x_prev, res - res_prev)
    end

    @test f(x) <= f_star + (1 + abs(f_star)) * 1e-6

end
