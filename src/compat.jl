if VERSION < v"1.1"
    using LinearAlgebra
    LinearAlgebra.mul!(C::AbstractVecOrMat, J::UniformScaling, B::AbstractVecOrMat) = mul!(C, J.λ, B)
end
