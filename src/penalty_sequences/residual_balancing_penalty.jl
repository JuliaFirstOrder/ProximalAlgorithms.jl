"""
	ResidualBalancingPenalty{R,T}

Adaptive penalty parameter strategy based on He et al.'s method. Updates penalties based on 
the ratio of primal and dual residuals to keep primal and dual residuals within a factor of
each other.

# Arguments
- `rho::R`: Initial penalty parameters (will be set if empty)
- `mu::T=10.0`: Residual ratio threshold
- `tau::T=2.0`: Penalty update factor
- `normalized::Bool=false`: Whether to normalize residuals by their respective tolerances before comparison.
- `adp_freq::Int=1`: Frequency of adaptation (every adp_freq iterations)
- `adp_start_iter::Int=1`: Iteration to start adaptation
- `adp_end_iter::Int=typemax(Int)`: Iteration to end adaptation
- `current_iter::Int=0`: Current iteration counter

# References
1. He, B. S., Yang, H., & Wang, S. L. (2000). "Alternating direction method with
   self-adaptive penalty parameters for monotone variational inequalities."
   Journal of Optimization Theory and Applications, 106(2), 337-356.
2. Boyd , S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
   "Distributed optimization and statistical learning via the alternating direction method of multipliers."
   Foundations and Trends in Machine Learning, 3(1), 1-122.
3. Lozenski, L., McCann, M. T., & Wohlberg, B. (2025). "An Adaptive Multiparameter
   Penalty Selection Method for Multiconstraint and Multiblock ADMM (No. arXiv:2502.21202).
   arXiv. https://doi.org/10.48550/arXiv.2502.21202
"""
@kwdef mutable struct ResidualBalancingPenalty{R,T} <: PenaltySequence
	rho::R = nothing
	mu::T = 10.0
	tau::T = 2.0
    normalized::Bool = false
	adp_freq::Int = 1
	adp_start_iter::Int = 2
	adp_end_iter::Int = typemax(Int)
	current_iter::Int = 0
	function ResidualBalancingPenalty{R,T}(
		rho::R,
		mu::T,
		tau::T,
        normalized::Bool,
		adp_freq::Int,
		adp_start_iter::Int,
		adp_end_iter::Int,
		current_iter::Int,
	) where {R,T}
		@assert adp_start_iter >= 2
		@assert adp_start_iter <= adp_end_iter
		@assert adp_freq > 0
		@assert current_iter >= 0
		new{R,T}(
			isnothing(rho) ? nothing : copy(rho),
			mu,
			tau,
            normalized,
			adp_freq,
			adp_start_iter,
			adp_end_iter,
			current_iter,
		)
	end
end

# Constructors
function ResidualBalancingPenalty(rho::R, mu::T, args...) where {R,T}
	ResidualBalancingPenalty{R,T}(rho, mu, args...)
end
function ResidualBalancingPenalty(rho::Union{AbstractVector,Number}; kwargs...)
	ResidualBalancingPenalty(; rho=rho, kwargs...)
end

function reinstantiate_penalty_sequence(
	seq::ResidualBalancingPenalty, ::Type{R}, rho
) where {R}
	final_rho = ensure_correct_value(seq.rho, R, rho)
	ResidualBalancingPenalty(;
		rho=final_rho,
		mu=R(seq.mu),
		tau=R(seq.tau),
		adp_freq=seq.adp_freq,
		adp_start_iter=seq.adp_start_iter,
		adp_end_iter=seq.adp_end_iter,
		current_iter=0,
	)
end

function get_next_rho!(seq::ResidualBalancingPenalty, ::ADMMIteration, state::ADMMState)
	seq.current_iter += 1
	changed = false
	if check_iter(seq)
		for i in eachindex(seq.rho)
            rᵏ_norm = state.rᵏ_norm[i]
            sᵏ_norm = state.sᵏ_norm[i]
            # If normalized, scale residuals to ensure they are non-negative
			if seq.normalized
				rᵏ_norm /= state.ϵᵖʳⁱ[i]
				sᵏ_norm /= state.ϵᵈᵘᵃ[i]
			end
			if rᵏ_norm > seq.mu * sᵏ_norm
				seq.rho[i] *= seq.tau
                state.u[i] ./= seq.tau
				changed = true
			elseif sᵏ_norm > seq.mu * rᵏ_norm
				seq.rho[i] /= seq.tau
                state.u[i] .*= seq.tau
				changed = true
			end
		end
	end
	return seq.rho, changed
end
