"""
    SpectralRadiusBoundPenalty{R,T}

Adaptive penalty parameter strategy based on spectral radius bound. Updates penalties using the formula:
    ρ = ‖yᵢ‖ / ‖zᵢ‖

# Arguments
- `rho::R`: Initial penalty parameters (one per regularizer block)
- `tau::T=10`: Scaling factor for the penalty update (default is 10)
- `eta::T=100`: Negative exponent of damping factor ωₙ := 2ˢ where s = -n/eta (default is 100)
- `adp_freq::Int=1`: Frequency of adaptation (every adp_freq iterations)
- `adp_start_iter::Int=2`: Iteration to start adaptation
- `adp_end_iter::Int=typemax(Int)`: Iteration to end adaptation
- `current_iter::Int=0`: Current iteration counter

# References
1. Lorenz, D. A., & Tran-Dinh, Q. (2019). Non-stationary Douglas–Rachford and
   alternating direction method of multipliers: Adaptive step-sizes and convergence.
   Computational Optimization and Applications, 74(1), 67–92. https://doi.org/10.1007/s10589-019-00106-9
2. Lozenski, L., McCann, M. T., & Wohlberg, B. (2025). An Adaptive Multiparameter
   Penalty Selection Method for Multiconstraint and Multiblock ADMM (No. arXiv:2502.21202).
   arXiv. https://doi.org/10.48550/arXiv.2502.21202
"""
@kwdef mutable struct SpectralRadiusBoundPenalty{R,T,T2} <: PenaltySequence
    rho::R = nothing
    tau::T = nothing
    eta::T2 = nothing
    adp_freq::Int = 1
    adp_start_iter::Int = 2
    adp_end_iter::Int = typemax(Int)
    current_iter::Int = 0
    function SpectralRadiusBoundPenalty{R,T,T2}(
        rho::R,
        tau::T,
        eta::T2,
        adp_freq::Int,
        adp_start_iter::Int,
        adp_end_iter::Int,
        current_iter::Int
    ) where {R,T,T2}
        @assert adp_start_iter >= 2
        @assert adp_start_iter <= adp_end_iter
        @assert adp_freq > 0
        @assert current_iter >= 0
        new{R,T,T2}(
            isnothing(rho) ? nothing : copy(rho),
            isnothing(tau) ? nothing : copy(tau),
            isnothing(eta) ? nothing : copy(eta),
            adp_freq,
            adp_start_iter,
            adp_end_iter,
            current_iter
        )
    end
end

# Constructors
function SpectralRadiusBoundPenalty(rho::R, tau::T, eta::T2, args...) where {R,T,T2}
    SpectralRadiusBoundPenalty{R,T,T2}(rho, tau, eta, args...)
end
function SpectralRadiusBoundPenalty(rho::Union{AbstractVector,Number}; kwargs...)
    SpectralRadiusBoundPenalty(; rho=rho, kwargs...)
end

function reinstantiate_penalty_sequence(
    seq::SpectralRadiusBoundPenalty, ::Type{R}, rho
) where {R}
    final_rho = ensure_correct_value(seq.rho, R, rho)
	n_blocks = length(final_rho)
    default_tau = fill(R(10.0), n_blocks)
    tau_vec = ensure_correct_value(seq.tau, R, default_tau)
	default_eta = fill(R(100.0), n_blocks)
	eta_vec = ensure_correct_value(seq.eta, R, default_eta)
    T = typeof(final_rho)
    SpectralRadiusBoundPenalty{T,T,T}(;
        rho=final_rho,
        tau=tau_vec,
        eta=eta_vec,
        adp_freq=seq.adp_freq,
        adp_start_iter=seq.adp_start_iter,
        adp_end_iter=seq.adp_end_iter,
        current_iter=0
    )
end

function get_next_rho!(
    seq::SpectralRadiusBoundPenalty, iter::ADMMIteration, state::ADMMState
)
    seq.current_iter += 1

    if mod(seq.current_iter - seq.adp_start_iter, seq.adp_freq) == 0 &&
        seq.adp_start_iter < seq.current_iter < seq.adp_end_iter
        changed = false
        for i in eachindex(iter.g)
            # Current penalty parameter for this block
            ρ, τ, η = seq.rho[i], seq.tau[i], seq.eta[i]

            # Spectral radius bound
            y_norm = ρ * norm(state.u[i])
            z_norm = norm(state.z[i])

            if y_norm ≈ 0 && z_norm > 0
                ρ /= τ
            elseif y_norm > 0 && z_norm ≈ 0
                ρ *= τ
            elseif y_norm > 0 && z_norm > 0
                n = seq.current_iter # - seq.adp_start_iter) ÷ seq.adp_freq
                ω = 2^(-n / η)
                ρ = (1 - ω) * ρ + ω * y_norm / z_norm
            end # if y_norm ≈ 0 && z_norm ≈ 0 -> ρ remains unchanged

            if ρ != seq.rho[i]
                state.u[i] .*= seq.rho[i] / ρ
                seq.rho[i] = ρ
                changed = true
            end
        end
        return seq.rho, changed
    end

    return seq.rho, false
end
