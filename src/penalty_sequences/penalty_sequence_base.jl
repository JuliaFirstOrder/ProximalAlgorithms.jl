abstract type PenaltySequence end

"""
	reinstantiate_penalty_sequence(seq::PenaltySequence, ::Type{R}, rho) where {R}

Reinstantiate a penalty sequence to ensure it has the correct type and missing penalty parameters
can also be set. This is useful to free the user from having to manually convert types when
specifying penalty sequences and their parameters, and enforce type consistency automatically by
this function.
"""
function reinstantiate_penalty_sequence end

function ensure_correct_value(old_value, ::Type{R}, new_value) where {R}
	final_rho = isnothing(new_value) ? old_value : new_value
	if final_rho isa Number
		final_rho = fill(final_rho, length(new_value))
	end
	return R.(final_rho)  # Ensure correct type conversion
end

"""
    check_iter(seq::PenaltySequence)

Check if the penalty sequence should update based on its current iteration, start and end iteration,
and adaptation frequency.

# Arguments
- `seq::PenaltySequence`: The penalty sequence object containing iteration and adaptation parameters.

# Returns
- `Bool`: `true` if the penalty sequence should update, `false` otherwise.
"""
function check_iter(seq::PenaltySequence)::Bool
    return mod(seq.current_iter - seq.adp_start_iter, seq.adp_freq) == 0 &&
           seq.adp_start_iter <= seq.current_iter < seq.adp_end_iter
end
