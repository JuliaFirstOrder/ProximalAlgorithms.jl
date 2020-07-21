using Test

@testset "Nesterov accel. ($R)" for R in [Float32, Float64]

using LinearAlgebra
using ProximalAlgorithms: NesterovAcceleration, update!

n, k = 20, 10

Q = randn(R, n, k)
H = Q*Q' + I
l = randn(R, n)

f(x) = dot(x, H*x)/2 + dot(x, l)

x_star = -H\l
f_star = f(x_star)
norm_x_star = norm(x_star)

Lip = opnorm(H)
gamma = 1/Lip
x = zeros(n)
res = zeros(n)
x_prev = zeros(n)
res_prev = zeros(n)

acc = NesterovAcceleration(x)

mul!(res, H, x)
res .+= l
res .*= gamma

d = acc * res

for it in 1:500
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

    # test that iterates satisfy Thm 4.4 from Beck, Teboulle (2009)
    @test f(x) - f_star <= 2/(gamma * (it+1)^2) * norm_x_star^2
end

end
