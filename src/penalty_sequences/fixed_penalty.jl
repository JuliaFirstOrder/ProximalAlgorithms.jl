"""
	FixedPenalty{R}

Fixed (non-adaptive) penalty parameters for ADMM. This is the simplest strategy where
penalty parameters remain constant throughout iterations.

# Arguments
- `rho::R`: Vector of fixed penalty parameters
"""
@kwdef struct FixedPenalty{R} <: PenaltySequence
	rho::R = nothing
end

# Constructors
FixedPenalty(rho::AbstractVector) = FixedPenalty{typeof(rho)}(collect(rho))
FixedPenalty(rho::Number) = FixedPenalty{typeof(rho)}(rho)

function reinstantiate_penalty_sequence(seq::FixedPenalty, ::Type{R}, rho) where {R}
	final_rho = ensure_correct_value(seq.rho, R, rho)
	FixedPenalty(final_rho)
end

function get_next_rho!(seq::FixedPenalty, ::ADMMIteration, ::ADMMState)
	seq.rho, false
end