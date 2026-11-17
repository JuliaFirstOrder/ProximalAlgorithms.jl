"""
    SpectralRadiusApproximationPenalty{R,T}

Adaptive penalty parameter strategy based on spectral radius approximation. Updates penalties using the formula:
    ρ = ‖yᵢ - yᵢ₋₁‖ / ‖(zᵢ - zᵢ₋₁)‖

# Arguments
- `rho::R`: Initial penalty parameters (one per regularizer block)
- `tau::T=10`: Scaling factor for the penalty update (default is 10)
- `adp_freq::Int=1`: Frequency of adaptation (every adp_freq iterations)
- `adp_start_iter::Int=2`: Iteration to start adaptation
- `adp_end_iter::Int=typemax(Int)`: Iteration to end adaptation
- `current_iter::Int=0`: Current iteration counter

# References
1. Mccann, M. T., & Wohlberg, B. (2024). Robust and Simple ADMM Penalty Parameter Selection.
   IEEE Open Journal of Signal Processing, 5, 402–420.
   https://doi.org/10.1109/OJSP.2023.3349115
2. Lozenski, L., McCann, M. T., & Wohlberg, B. (2025). An Adaptive Multiparameter
   Penalty Selection Method for Multiconstraint and Multiblock ADMM (No. arXiv:2502.21202).
   arXiv. https://doi.org/10.48550/arXiv.2502.21202
"""
@kwdef mutable struct SpectralRadiusApproximationPenalty{R,T} <: PenaltySequence
    rho::R = nothing
    tau::T = nothing
    adp_freq::Int = 1
    adp_start_iter::Int = 2
    adp_end_iter::Int = typemax(Int)
    current_iter::Int = 0
    uᵢ₋₁::Union{Nothing,Tuple} = nothing  # Storage for previous u values
    function SpectralRadiusApproximationPenalty{R,T}(
        rho::R,
        tau::T,
        adp_freq::Int,
        adp_start_iter::Int,
        adp_end_iter::Int,
        current_iter::Int,
        uᵢ₋₁::Union{Nothing,Tuple}
    ) where {R,T}
        @assert adp_start_iter >= 2
        @assert adp_start_iter <= adp_end_iter
        @assert adp_freq > 0
        @assert current_iter >= 0
        new{R,T}(
            isnothing(rho) ? nothing : copy(rho),
            isnothing(tau) ? nothing : copy(tau),
            adp_freq,
            adp_start_iter,
            adp_end_iter,
            current_iter,
            uᵢ₋₁
        )
    end
end

# Constructors
function SpectralRadiusApproximationPenalty(rho::R, tau::T, args...) where {R,T}
    SpectralRadiusApproximationPenalty{R,T}(rho, tau, args...)
end
function SpectralRadiusApproximationPenalty(rho::Union{AbstractVector,Number}; kwargs...)
    SpectralRadiusApproximationPenalty(; rho=rho, kwargs...)
end

function reinstantiate_penalty_sequence(
    seq::SpectralRadiusApproximationPenalty, ::Type{R}, rho
) where {R}
    final_rho = ensure_correct_value(seq.rho, R, rho)
	n_blocks = length(final_rho)
	default_tau = fill(R(10.0), n_blocks)
	tau_vec = ensure_correct_value(seq.tau, R, default_tau)
    T = typeof(final_rho)
    SpectralRadiusApproximationPenalty{T,T}(;
        rho=final_rho,
        tau=tau_vec,
        adp_freq=seq.adp_freq,
        adp_start_iter=seq.adp_start_iter,
        adp_end_iter=seq.adp_end_iter,
        current_iter=0,
        uᵢ₋₁=nothing
    )
end

function get_next_rho!(
    seq::SpectralRadiusApproximationPenalty, iter::ADMMIteration, state::ADMMState
)
    seq.current_iter += 1

    # Initialize storage _after_ first iteration
    if seq.current_iter == max(2, seq.adp_start_iter - seq.adp_freq)
        seq.uᵢ₋₁ = Tuple(copy.(state.u))
        return seq.rho, false
    elseif 2 < seq.current_iter && check_iter(seq)
        changed = false
        for i in eachindex(iter.g)
            # Current penalty parameter for this block
            ρ, τ = seq.rho[i], seq.tau[i]

            # Spectral radius approximation
            temp = seq.uᵢ₋₁[i]
            Δy = @. temp = state.u[i] - seq.uᵢ₋₁[i]
            Δy_norm = ρ * norm(Δy)
            Δz = @. temp = state.z[i] - state.z_old[i]
            Δz_norm = norm(Δz)

            if Δy_norm ≈ 0 && Δz_norm > 0
                ρ /= τ
            elseif Δy_norm > 0 && Δz_norm ≈ 0
                ρ *= τ
            elseif Δy_norm > 0 && Δz_norm > 0
                ρ = Δy_norm / Δz_norm
            end # if Δy_norm ≈ 0 && Δz_norm ≈ 0 -> ρ remains unchanged

            if ρ != seq.rho[i]
                state.u[i] .*= seq.rho[i] / ρ
                seq.rho[i] = ρ
                changed = true
            end

            # Update storage for next iteration
            seq.uᵢ₋₁[i] .= state.u[i]
        end
        return seq.rho, changed
    end

    return seq.rho, false
end
