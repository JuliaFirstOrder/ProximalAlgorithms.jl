using ProximalOperators
using ProximalOperators: Zero

ProximalOperators.Conjugate(f::Zero) = IndZero()
ProximalOperators.Conjugate(f::IndZero) = Zero()
ProximalOperators.Conjugate(f::SqrNormL2) = SqrNormL2(1.0/f.lambda)

# TODO: Add other useful functions and calculus rules such as translation
