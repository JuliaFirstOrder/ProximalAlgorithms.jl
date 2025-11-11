mutable struct AndersonAccelerationOperator{M,I,T}
    currmem::I
    curridx::I
    s::T
    y::T
    s_M::Vector{T}
    y_M::Vector{T}
end

function AndersonAccelerationOperator{M}(x::T) where {M,T}
    s_M = [zero(x) for i = 1:M]
    y_M = [zero(x) for i = 1:M]
    s = zero(x)
    y = zero(x)
    AndersonAccelerationOperator{M,typeof(0),T}(0, 0, s, y, s_M, y_M)
end

AndersonAccelerationOperator(M, x) = AndersonAccelerationOperator{M}(x)

function update!(L::AndersonAccelerationOperator{M,I,T}, s, y) where {M,I,T}
    L.s .= s
    L.y .= y
    L.curridx += 1
    if L.curridx > M
        L.curridx = 1
    end
    L.currmem += 1
    if L.currmem > M
        L.currmem = M
    end
    copyto!(L.s_M[L.curridx], L.s)
    copyto!(L.y_M[L.curridx], L.y)
    return L
end

function reset!(L::AndersonAccelerationOperator{M,I,T}) where {M,I,T}
    L.currmem, L.curridx = zero(I), zero(I)
end

function (*)(L::AndersonAccelerationOperator, v)
    w = similar(v)
    return mul!(w, L, v)
end

function mul!(d::T, L::AndersonAccelerationOperator{M,I,T}, v::T) where {M,I,T}
    if L.currmem == 0
        d .= v
    else
        # TODO: optimize this
        S = hcat(L.s_M[1:L.currmem]...)
        Y = hcat(L.y_M[1:L.currmem]...)
        # H = I + (S - Y)inv(Y'Y)Y'
        d .= v .+ (S - Y) * (pinv(Y' * Y) * (Y' * v))
    end
    return d
end

struct AndersonAcceleration{M} end

AndersonAcceleration(M) = AndersonAcceleration{M}()

acceleration_style(::Type{<:AndersonAcceleration}) = QuasiNewtonStyle()

function initialize(::AndersonAcceleration{M}, x) where {M}
    return AndersonAccelerationOperator{M}(x)
end
