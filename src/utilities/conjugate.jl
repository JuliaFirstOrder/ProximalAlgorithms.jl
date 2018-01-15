import ProximalOperators: Conjugate

Conjugate(f::IndFree) = IndZero()
Conjugate(f::IndZero) = IndFree()
Conjugate(f::SqrNormL2) = SqrNormL2(1.0/f.lambda)

# TODO: Add other useful functions and calculus rules such as translation
