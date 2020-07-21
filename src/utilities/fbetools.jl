function f_model(
    f_x::R, grad_f_x::AbstractArray{C}, res::AbstractArray{C}, gamma::R
) where {R <: Real, C <: RealOrComplex{R}}
    return f_x - real(dot(grad_f_x, res)) + (0.5/gamma)*norm(res)^2
end
