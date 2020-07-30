using LinearAlgebra

mutable struct Broyden{R<:Real,C<:Union{R,Complex{R}},T<:AbstractArray{C}}
    H
    theta_bar::R
    function Broyden{R,C,T}(x::T, H, theta_bar) where {R,C,T}
        new(H, theta_bar)
    end
end

Broyden(
    x::T;
    H = I,
    theta_bar = R(0.2),
) where {R<:Real,C<:Union{R,Complex{R}},T<:AbstractArray{C}} =
    Broyden{R,C,T}(x, H, theta_bar)

_sign(x::R) where {R} = x == 0 ? R(1) : sign(x)

function update!(L::Broyden{R,C,T}, s, y) where {R,C,T}
    Hy = L.H * y
    sH = s' * L.H
    delta = dot(Hy, s) / norm(s)^2
    theta = if abs(delta) >= L.theta_bar
        R(1)
    else
        (1 - _sign(delta) * L.theta_bar) / (1 - delta)
    end
    L.H += (s - Hy) / dot(s, (1 / theta - 1) * s + Hy) * sH
end

import Base: *

function (*)(L::Broyden, v)
    w = similar(v)
    mul!(w, L, v)
end

import LinearAlgebra: mul!

mul!(d::T, L::Broyden, v::T) where {T} = mul!(d, L.H, v)
