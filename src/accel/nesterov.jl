using LinearAlgebra

mutable struct NesterovAcceleration{R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}}
    k::Int
    x_prev::T
    x_curr::T
end

function NesterovAcceleration(x::T) where {R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}}
    NesterovAcceleration{R, C, T}(0, zero(x), zero(x))
end

function update!(L::NesterovAcceleration{R, C, T}, x, r) where {R, C, T}
    L.x_prev .= L.x_curr
    L.x_curr .= x .- r
    L.k += 1
    return L
end

function reset!(L::NesterovAcceleration{R, C, T}) where {R, C, T}
    L.k = 0
end

import Base: *

function (*)(L::NesterovAcceleration, x)
    y = similar(x)
    mul!(y, L, x)
end

import LinearAlgebra: mul!

function mul!(d::T, L::NesterovAcceleration{R, C, T}, v::T) where {R, C, T}
    if L.k == 0
        d .= 0
        return
    elseif L.k == 1
        d .= v
        return
    end
    d .= v .- (L.k - 1)/(L.k + 2) .* (L.x_curr .- L.x_prev)
end
