@testset "Conjugate" begin

    using ProximalOperators
    using ProximalOperators: Zero
    using ProximalAlgorithms

    x = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]

    f = Conjugate(IndZero()) # = IndFree
    grad_f_x, f_x = gradient(f, x)
    @test iszero(grad_f_x)
    @test iszero(f_x)

    g = Conjugate(Zero()) # = IndZero
    prox_g_x, g_y = prox(g, x)
    @test iszero(prox_g_x)
    @test iszero(g_y)

    l = Conjugate(SqrNormL2())
    grad_l_x, l_x = gradient(l, x)
    @test isequal(grad_l_x, x)

    lam = 2
    l = Conjugate(SqrNormL2(lam))
    grad_l_x, l_x = gradient(l, x)
    @test isequal(grad_l_x, x/lam)
    
end
