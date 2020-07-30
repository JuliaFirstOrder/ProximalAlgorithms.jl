using LinearAlgebra

mutable struct NesterovAcceleration{R<:Real,C<:Union{R,Complex{R}},T<:AbstractArray{C}}
    k::Int
    s::T
end

function NesterovAcceleration(
    x::T,
) where {R<:Real,C<:Union{R,Complex{R}},T<:AbstractArray{C}}
    NesterovAcceleration{R,C,T}(0, zero(x))
end

function update!(L::NesterovAcceleration{R,C,T}, s, y) where {R,C,T}
    L.s .= s
    L.k += 1
    return L
end

function reset!(L::NesterovAcceleration{R,C,T}) where {R,C,T}
    L.k = 0
end

import Base: *

function (*)(L::NesterovAcceleration, v)
    w = similar(v)
    mul!(w, L, v)
end

import LinearAlgebra: mul!

function mul!(d::T, L::NesterovAcceleration{R,C,T}, v::T) where {R,C,T}
    if L.k == 0
        d .= 0
    elseif L.k == 1
        d .= v
    else
        d .= v .- (L.k - 1) / (L.k + 2) .* L.s
    end
end
