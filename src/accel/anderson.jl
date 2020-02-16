using LinearAlgebra

mutable struct AndersonAcceleration{R <: Real, C <: Union{R, Complex{R}}, I <: Integer, T <: AbstractArray{C}, M}
    currmem::I
    curridx::I
    s::T
    y::T
    s_M::Vector{T}
    y_M::Vector{T}
end

function AndersonAcceleration(x::T, M::I) where {R <: Real, C <: Union{R, Complex{R}}, I <: Integer, T <: AbstractArray{C}}
    s_M = [zero(x) for i = 1:M]
    y_M = [zero(x) for i = 1:M]
    s = zero(x)
    y = zero(x)
    AndersonAcceleration{R, C, I, T, M}(0, 0, s, y, s_M, y_M)
end

function update!(L::AndersonAcceleration{R, C, I, T, M}, s, y) where {R, C, I, T, M}
    L.s .= s
    L.y .= y
    L.curridx += 1
    if L.curridx > M L.curridx = 1 end
    L.currmem += 1
    if L.currmem > M L.currmem = M end
    copyto!(L.s_M[L.curridx], L.s)
    copyto!(L.y_M[L.curridx], L.y)
    return L
end

function reset!(L::AndersonAcceleration{R, C, I, T, M}) where {R, C, I, T, M}
    L.currmem, L.curridx = zero(I), zero(I)
end

import Base: *

function (*)(L::AndersonAcceleration, v)
    w = similar(v)
    mul!(w, L, v)
end

import LinearAlgebra: mul!

function mul!(d::T, L::AndersonAcceleration{R, C, I, T, M}, v::T) where {R, C, I, T, M}
    if L.currmem == 0
        d .= v
    else
        # TODO: optimize this
        S = hcat(L.s_M[1:L.currmem]...)
        Y = hcat(L.y_M[1:L.currmem]...)
        # H = I + (S - Y)inv(Y'Y)Y'
        d .= v .+ (S - Y)*(pinv(Y'*Y)*(Y'*v))
    end
end
