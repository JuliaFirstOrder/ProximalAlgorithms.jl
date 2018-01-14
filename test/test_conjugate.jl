x = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
f = Conjugate(IndZero()) # = IndFree
grad_f_x, f_x = gradient(f, x)
@test iszero(grad_f_x)
@test iszero(f_x)
g = Conjugate(IndFree()) # = IndZero
prox_g_x, g_y = prox(g, x)
@test iszero(prox_g_x)
@test iszero(g_y)
