import Base: A_mul_B!, Ac_mul_B!, *, transpose, size

struct Identity
    size::Tuple
end

size(A::Identity) = (A.size, A.size)
size(A::Identity, dim::Integer) = A.size

A_mul_B!(y, A::Identity, x) = blockcopy!(y, x)
Ac_mul_B!(y, A::Identity, x) = A_mul_B!(y, A, x)

transpose(A::Identity) = A

(*)(A::Identity, x) = blockcopy(x)
