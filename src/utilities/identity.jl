import Base: *, size
import LinearAlgebra: adjoint, opnorm, mul!

struct Identity
    size::Tuple
end

size(A::Identity) = (A.size, A.size)
size(A::Identity, dim::Integer) = A.size

mul!(y, A::Identity, x) = y .= x

adjoint(A::Identity) = A

(*)(A::Identity, x) = copy(x)

opnorm(A::Identity) = 1.0
opnorm(A::Identity, p::Real) = 1.0

