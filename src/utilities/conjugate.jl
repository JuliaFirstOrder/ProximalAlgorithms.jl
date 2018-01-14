import ProximalOperators: Conjugate

Conjugate(f::IndFree) = IndZero()
Conjugate(f::IndZero) = IndFree()
