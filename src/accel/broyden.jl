using LinearAlgebra
import Base: *
import LinearAlgebra: mul!

mutable struct BroydenOperator{R<:Real,C<:Union{R,Complex{R}},T<:AbstractArray{C}}
    H
    theta_bar::R
    function BroydenOperator{R,C,T}(x::T, H, theta_bar) where {R,C,T}
        new(H, theta_bar)
    end
end

BroydenOperator(
    x::T;
    H = I,
    theta_bar = R(0.2),
) where {R<:Real,C<:Union{R,Complex{R}},T<:AbstractArray{C}} =
    BroydenOperator{R,C,T}(x, H, theta_bar)

_sign(x::R) where {R} = x == 0 ? R(1) : sign(x)

function update!(L::BroydenOperator{R,C,T}, s, y) where {R,C,T}
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

function reset!(L::BroydenOperator{R,C,T}) where {R,C,T}
    L.H = I
    L.theta_bar = R(0.2)
end

function (*)(L::BroydenOperator, v)
    w = similar(v)
    return mul!(w, L, v)
end

function mul!(d::T, L::BroydenOperator, v::T) where {T}
    mul!(d, L.H, v)
    return d
end

Base.@kwdef struct Broyden{R}
    theta_bar::R = 0.2
end

acceleration_style(::Type{<:Broyden}) = QuasiNewtonStyle()

function initialize(broyden::Broyden, x)
    return BroydenOperator(x, theta_bar=broyden.theta_bar)
end
