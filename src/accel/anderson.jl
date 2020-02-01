using LinearAlgebra

mutable struct AndersonAcceleration{R <: Real, C <: Union{R, Complex{R}}, I <: Integer, T <: AbstractArray{C}, M}
    currmem::I
    curridx::I
    x_prev::Union{T, Nothing}
    r_prev::Union{T, Nothing}
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
    AndersonAcceleration{R, C, I, T, M}(0, 0, nothing, nothing, s, y, s_M, y_M)
end

function update!(L::AndersonAcceleration{R, C, I, T, M}, x, r) where {R, C, I, T, M}
    if L.x_prev === nothing || L.r_prev === nothing
        L.x_prev = copy(x)
        L.r_prev = copy(r)
        return L
    end
    L.s .= x .- L.x_prev
    L.y .= r .- L.r_prev
    L.curridx += 1
    if L.curridx > M L.curridx = 1 end
    L.currmem += 1
    if L.currmem > M L.currmem = M end
    copyto!(L.s_M[L.curridx], L.s)
    copyto!(L.y_M[L.curridx], L.y)
    L.x_prev .= x
    L.r_prev .= r
    return L
end

function reset!(L::AndersonAcceleration{R, C, I, T, M}) where {R, C, I, T, M}
    L.currmem, L.curridx = zero(I), zero(I)
    L.x_prev, L.r_prev = nothing, nothing
end

import Base: *

function (*)(L::AndersonAcceleration, x)
    y = similar(x)
    mul!(y, L, x)
end

import LinearAlgebra: mul!

function mul!(d::T, L::AndersonAcceleration{R, C, I, T, M}, v::T) where {R, C, I, T, M}
    if L.currmem == 0
        d .= v
        return
    end
    # TODO: optimize this
    S = hcat(L.s_M[1:L.currmem]...)
    Y = hcat(L.y_M[1:L.currmem]...)
    # H = I + (S - Y)inv(Y'Y)Y'
    d .= v .+ (S - Y)*(pinv(Y'*Y)*(Y'*v))
end
