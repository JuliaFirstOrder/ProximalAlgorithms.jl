import Base: A_mul_B!, Ac_mul_B!, *, transpose, size

# function A_mul_B!(y::Tuple, A::UniformScaling{T}, x::Tuple) where T
#     if A.λ == one(T)
#         blockcopy!(y, x)
#     else
#         blockaxpy!(y, 0.0, A.λ, x)
#     end
# end

# Ac_mul_B!(y::Tuple, A::UniformScaling, x::Tuple) = A_mul_B!(y, A, x)

struct Identity
    size::Tuple
end

size(A::Identity) = (A.size, A.size)
size(A::Identity, dim::Integer) = A.size

A_mul_B!(y, A::Identity, x) = blockcopy!(y, x)
Ac_mul_B!(y, A::Identity, x) = A_mul_B!(y, A, x)

transpose(A::Identity) = A

(*)(A::Identity, x) = blockcopy(x)