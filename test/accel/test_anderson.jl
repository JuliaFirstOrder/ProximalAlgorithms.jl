using Test
using LinearAlgebra
using ProximalAlgorithms: AndersonAcceleration, initialize, AndersonAccelerationOperator, update!, reset!

@testset "Anderson accel. ($R)" for R in [Float32, Float64]
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
    grad_f(x) = H * x + l

    x_star = -H \ l
    f_star = f(x_star)

    Lip = opnorm(H)
    gamma = 1 / Lip
    x = zeros(R, n)

    @inferred initialize(AndersonAcceleration(5), x)
    
    acc = AndersonAccelerationOperator(5, x)

    grad_f_x = grad_f(x)

    for it = 1:10
        d = @inferred acc * grad_f_x
        x = x - d

        grad_f_x_prev = grad_f_x
        grad_f_x = grad_f(x)

        update!(acc, -d, grad_f_x - grad_f_x_prev)
    end

    @test f(x) <= f_star + (1 + abs(f_star)) * sqrt(eps(R))

    reset!(acc)

    @test acc * x == x
end
