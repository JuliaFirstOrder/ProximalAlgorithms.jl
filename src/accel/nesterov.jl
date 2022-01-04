struct FixedNesterovSequence{R} end

"""
    FixedNesterovSequence(R::Type)

Construct the standard sequence of extrapolation coefficients, of type `R`, used in Nesterov-accelerated gradient methods.

Argument `R` should be a floating point type, and specifies the type of the elements in the sequence.

See also: [`SimpleNesterovSequence`](@ref), [`ConstantNesterovSequence`](@ref).
"""
FixedNesterovSequence(R) = FixedNesterovSequence{R}()

function Base.iterate(::FixedNesterovSequence{R}, t=R(1)) where R
    t_next = (1 + sqrt(1 + 4 * t^2)) / 2
    return (t - 1) / t_next, t_next
end

Base.IteratorSize(::Type{<:FixedNesterovSequence}) = Base.IsInfinite()
Base.IteratorEltype(::Type{FixedNesterovSequence{R}}) where R = R

struct SimpleNesterovSequence{R} end

"""
    SimpleNesterovSequence(R::Type)

Construct an iterator yielding the sequence `(k - 1) / (k + 2)`, of type `R`, for `k >= 1`.

This is the "simplified" sequence of extrapolation coefficients used in Nesterov-accelerated gradient methods.
Argument `R` should be a floating point type, and specifies the type of the elements in the sequence.

See also: [`FixedNesterovSequence`](@ref), [`ConstantNesterovSequence`](@ref).
"""
SimpleNesterovSequence(R) = SimpleNesterovSequence{R}()

Base.iterate(::SimpleNesterovSequence{R}, k=1) where {R} = R(k - 1) / (k + 2), k + 1

Base.IteratorSize(::Type{<:SimpleNesterovSequence}) = Base.IsInfinite()
Base.IteratorEltype(::Type{SimpleNesterovSequence{R}}) where R = R

"""
    ConstantNesterovSequence(m::R, stepsize::R)

Construct a constant iterator yielding `(1 - sqrt(m * stepsize)) / (1 + sqrt(m * stepsize))` with type `R`.

This sequence of extrapolation coefficients is commonly used in Nesterov-accelerated gradient methods
when applied to strongly convex functions.

See also: [`FixedNesterovSequence`](@ref), [`SimpleNesterovSequence`](@ref).
"""
function ConstantNesterovSequence(m::R, stepsize::R) where R
    k_inverse = m * stepsize
    return repeated((1 - sqrt(k_inverse)) / (1 + sqrt(k_inverse)))
end

mutable struct AdaptiveNesterovSequence{R}
    m::R
    stepsize::R
    theta::R
end

"""
    AdaptiveNesterovSequence(m::R)

Construct an object that yields a sequence of extrapolation coefficients, commonly used in Nesterov-accelerated gradient methods.

Argument `m` is usually the convexity modulus of the interested function, so it should be zero for barely convex functions,
and greater than zero for strongly convex ones. The extrapolation coefficients can be extracted by calling [`next!`](@ref) on the
object.

Note that this type, as opposed to e.g. [`FixedNesterovSequence`](@ref), is not a standard Julia iterator, in that it requires
the sequence of stepsizes (input to [`next!`](@ref)) in order to produce the output sequence.

See also: [`next!`](@ref).
"""
AdaptiveNesterovSequence(m::R) where R = AdaptiveNesterovSequence{R}(m, -R(1), -R(1))

"""
    next!(seq::AdaptiveNesterovSequence{R}, stepsize::R)

Push a new stepsize value into the sequence, and return the next extrapolation coefficient.

See also: [`AdaptiveNesterovSequence`](@ref).
"""
function next!(seq::AdaptiveNesterovSequence{R}, stepsize::R) where R
    if seq.stepsize < 0
        seq.stepsize = stepsize
        seq.theta = seq.m > 0 ? sqrt(seq.m * stepsize) : R(1)
    end
    b = seq.theta^2 / seq.stepsize - seq.m
    delta = b^2 + 4 * (seq.theta^2) / (seq.stepsize * stepsize)
    theta = stepsize * (-b + sqrt(delta)) / 2
    beta = stepsize * (seq.theta) * (1 - seq.theta) / (seq.stepsize * theta + stepsize * seq.theta ^ 2)
    seq.stepsize = stepsize
    seq.theta = theta
    return beta
end

struct NesterovExtrapolation{S} end

NesterovExtrapolation(S) = NesterovExtrapolation{S}()
NesterovExtrapolation() = NesterovExtrapolation{SimpleNesterovSequence}()

acceleration_style(::Type{<:NesterovExtrapolation}) = NesterovStyle()

initialize(::NesterovExtrapolation{S}, x) where S = Iterators.Stateful(S{real(eltype(x))}())
