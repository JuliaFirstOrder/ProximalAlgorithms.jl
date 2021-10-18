using Test

@testset "Broyden ($R)" for R in [Float32, Float64]
    using LinearAlgebra
    using ProximalAlgorithms: Broyden, update!

    H = R[
        0.63287    0.330934   -0.156908   -0.294776    0.10761;
        0.330934   0.673201    0.0459778   0.231011   -0.235265;
       -0.156908   0.0459778   0.635812   -0.232261   -0.388775;
       -0.294776   0.231011   -0.232261    0.726854   -0.0691783;
        0.10761   -0.235265   -0.388775   -0.0691783   0.336262;
    ]
    l = R[1.0, 2.0, 3.0, 4.0, 5.0]

    n = length(l)

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

    acc = Broyden(x)

    mul!(res, H, x)
    res .+= l
    res .*= gamma

    d = acc * res

    for it = 1:10
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

    @test f(x) <= f_star + (1 + abs(f_star)) * sqrt(eps(R))

end
