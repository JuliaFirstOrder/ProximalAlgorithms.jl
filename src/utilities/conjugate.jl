import ProximalOperators: Conjugate

Conjugate(f::IndFree) = IndZero()
Conjugate(f::IndZero) = IndFree()
function Conjugate(f::SqrNormL2)
    return SqrNormL2(1.0/f.lambda)
end

#TODO: Add other useful functions and calculus rules such as translation

