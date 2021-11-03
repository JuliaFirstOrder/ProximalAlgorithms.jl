struct SimpleNesterovSequence{R} end

function Base.iterate(::SimpleNesterovSequence{R}, k=R(0)) where R
    val = iszero(k) ? R(0) : (k - 1) / (k + 2)
    return val, k + 1
end

struct NesterovSequence{R} end

function Base.iterate(::NesterovSequence{R}, thetas=(R(1), R(1))) where R
    theta_prev, theta = thetas
    return (theta_prev - 1) / theta, (theta, (1 + sqrt(1 + 4 * theta^2)) / 2)
end

struct NesterovExtrapolation{S} end

NesterovExtrapolation(S) = NesterovExtrapolation{S}()
NesterovExtrapolation() = NesterovExtrapolation{SimpleNesterovSequence}()

acceleration_style(::Type{<:NesterovExtrapolation}) = NesterovStyle()

initialize(::NesterovExtrapolation{S}, x) where S = Iterators.Stateful(S{real(eltype(x))}())
