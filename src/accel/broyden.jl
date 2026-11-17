struct BroydenOperator{R,TH}
    H::TH
    theta_bar::R
end

function BroydenOperator(::Type{T}, n::Integer, theta_bar = T(0.2)) where {T}
    H = Matrix{T}(I, n, n)
    BroydenOperator(H, theta_bar)
end

function BroydenOperator(x::AbstractVector{T}, theta_bar = T(0.2)) where {T}
    BroydenOperator(T, length(x), theta_bar)
end

_sign(x::R) where {R} = x == 0 ? R(1) : sign(x)

function update!(L::BroydenOperator{R,TH}, s, y) where {R,TH}
    Hy = L.H * y
    sH = s' * L.H
    delta = dot(Hy, s) / norm(s)^2
    theta = if abs(delta) >= L.theta_bar
        R(1)
    else
        (1 - _sign(delta) * L.theta_bar) / (1 - delta)
    end
    L.H .+= (s - Hy) / dot(s, (1 / theta - 1) * s + Hy) * sH
end

function reset!(L::BroydenOperator{R,TH}) where {R,TH}
    L.H .= 0
    L.H[diagind(L.H)] .= 1
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

function initialize(broyden::Broyden, x::AbstractVector{R}) where {R}
    return BroydenOperator(x, broyden.theta_bar)
end
