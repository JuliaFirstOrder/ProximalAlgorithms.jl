# Reference implementation for Convolutional Basis Pursuit DeNoising by B. Wohlberg:
# https://github.com/pengbao7598/PWLS-CSCGR/blob/master/CSC/cbpdn.m

"""
	WohlbergPenalty{R,T}

Adaptive penalty parameter scheme based on Wohlberg's improved residual balancing method.
This method extends the classical Boyd residual balancing by adaptively adjusting the 
scaling factor τ individually for each block in each iteration, rather than keeping it fixed.

The key innovation is that each τᵢ is updated based on the residual balance of its 
corresponding block i, which helps prevent oscillations and provides more stable convergence.

# Arguments
- `rho::R`: Initial penalty parameters (will be set if empty)
- `mu::T=10.0`: Residual ratio threshold (same as classical method)
- `tau::Vector{T}`: Per-block adaptive scaling factors (one for each rho)
- `tau_max::T=100.0`: Maximum allowed scaling factor
- `normalized::Bool=false`: Whether to normalize residuals by their respective tolerances
- `adp_freq::Int=1`: Frequency of adaptation (every adp_freq iterations)
- `adp_start_iter::Int=1`: Iteration to start adaptation
- `adp_end_iter::Int=typemax(Int)`: Iteration to end adaptation
- `current_iter::Int=0`: Current iteration counter

Note: We use ξ=1 for simplicity. It should be good enough for most cases.

# References
1. Wohlberg, B. (2017). "ADMM penalty parameter selection by residual balancing."
   arXiv preprint arXiv:1704.06209.
"""
@kwdef mutable struct WohlbergPenalty{R,VT,T} <: PenaltySequence
	rho::R = nothing
	mu::T = 10.0
	tau::VT = nothing  # Per-block adaptive scaling factors (one for each rho)
	tau_max::T = 100.0
    normalized::Bool = true
	adp_freq::Int = 1
	adp_start_iter::Int = 2
	adp_end_iter::Int = typemax(Int)
	current_iter::Int = 0
	function WohlbergPenalty{R,VT,T}(
		rho::R,
		mu::T,
		tau::VT,
		tau_max::T,
        normalized::Bool,
		adp_freq::Int,
		adp_start_iter::Int,
		adp_end_iter::Int,
		current_iter::Int,
	) where {R,VT,T}
		@assert adp_start_iter >= 2
		@assert adp_start_iter <= adp_end_iter
		@assert adp_freq > 0
		@assert current_iter >= 0
		new{R,VT,T}(
			isnothing(rho) ? nothing : copy(rho),
			mu,
			isnothing(tau) ? nothing : copy(tau),
			tau_max,
            normalized,
			adp_freq,
			adp_start_iter,
			adp_end_iter,
			current_iter,
		)
	end
end

# Constructors
function WohlbergPenalty(rho::R, mu::T, tau::VT, args...) where {R,VT,T}
	WohlbergPenalty{R,VT,T}(rho, mu, tau, args...)
end
function WohlbergPenalty(rho::Union{AbstractVector,Number}; kwargs...)
	WohlbergPenalty(; rho=rho, kwargs...)
end

function reinstantiate_penalty_sequence(seq::WohlbergPenalty, ::Type{R}, rho) where {R}
	final_rho = ensure_correct_value(seq.rho, R, rho)
	n_blocks = length(final_rho)
	default_tau = fill(R(2.0), n_blocks)
	tau_vec = ensure_correct_value(seq.tau, R, default_tau)
	WohlbergPenalty{typeof(final_rho),typeof(tau_vec),R}(;
		rho=final_rho,
		mu=R(seq.mu),
		tau=tau_vec,
		tau_max=R(seq.tau_max),
        normalized=seq.normalized,
		adp_freq=seq.adp_freq,
		adp_start_iter=seq.adp_start_iter,
		adp_end_iter=seq.adp_end_iter,
		current_iter=0,
	)
end

function get_next_rho!(seq::WohlbergPenalty, ::ADMMIteration, state::ADMMState)
	seq.current_iter += 1
	changed = false

	# Wohlberg's adaptive residual balancing: adapt tau individually for each block
	# based on the residual balance of that specific block
	if check_iter(seq)
        for i in eachindex(seq.rho)
			rᵏ_norm = state.rᵏ_norm[i]
			sᵏ_norm = state.sᵏ_norm[i]

			# Wohlberg's per-block adaptive tau update
			if 1 ≤ sqrt(rᵏ_norm / sᵏ_norm) < seq.tau_max
				seq.tau[i] = sqrt(rᵏ_norm / sᵏ_norm)
			elseif 1/seq.tau_max ≤ sqrt(rᵏ_norm / sᵏ_norm) < 1
				seq.tau[i] = sqrt(sᵏ_norm / rᵏ_norm)
			end
			# Otherwise, tau[i] remains unchanged (residuals are reasonably balanced)

            if seq.normalized
                rᵏ_norm /= state.ϵᵖʳⁱ[i]
                sᵏ_norm /= state.ϵᵈᵘᵃ[i]
            end
			# Apply residual balancing with the current tau[i] value
			if sᵏ_norm ≠ 0 && rᵏ_norm > seq.mu * sᵏ_norm
				seq.rho[i] *= seq.tau[i]
                state.u[i] ./= seq.tau[i]
				changed = true
			elseif rᵏ_norm ≠ 0 && sᵏ_norm > seq.mu * rᵏ_norm
				seq.rho[i] /= seq.tau[i]
                state.u[i] .*= seq.tau[i]
				changed = true
			end
		end
	end
	return seq.rho, changed
end
