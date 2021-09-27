function f_model(
    f_x::R,
    grad_f_x::AbstractArray{C},
    res::AbstractArray{C},
    L::R,
) where {R<:Real,C<:RealOrComplex{R}}
    return f_x - real(dot(grad_f_x, res)) + (L / 2) * norm(res)^2
end
