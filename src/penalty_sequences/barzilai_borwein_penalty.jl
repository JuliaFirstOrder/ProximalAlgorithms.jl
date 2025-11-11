# TODO: Probably something is wrong with this implementation, as it does not
#       converge in tests. Needs debugging.
# Original implementation by Z. Xu to be used for debugging:
#  - Adaptive ADMM: https://github.com/nightldj/admm_release/blob/master/2017-aistats-aadmm/solver/aadmm_core.m
#  - Adaptive Multiblock ADMM: https://github.com/nightldj/admm_release/blob/master/2019-thesis-amadmm/solver/amadmm_core.m

"""
	BarzilaiBorweinStorage{N,R,T}

Storage for intermediate variables used in Barzilai-Borwein spectral penalty parameter selection.

# Fields
- `uᵢ₋₁::Union{Nothing,AbstractArray}`: Dual variables from the previous iteration
- `ŷᵢ₋₁::Union{Nothing,AbstractArray}`: Hat dual variables from the previous iteration
- `xᵢ₋₁::Union{Nothing,AbstractArray}`: Primal variables from the previous iteration
- `zᵢ₋₁::Union{Nothing,AbstractArray}`: Primal variables from the previous iteration
- `temp::Union{Nothing,AbstractArray}`: Temporary storage for spectral computation
- `temp₂::Union{Nothing,AbstractArray}`: Additional temporary storage for spectral computation
- `temp₃::Union{Nothing,AbstractArray}`: Temporary storage for spectral computation (one per block)
"""
struct BarzilaiBorweinStorage{N,Ty,Tx}
	uᵢ₋₁::NTuple{N,Ty}
	ŷᵢ₋₁::NTuple{N,Ty}
	xᵢ₋₁::Tx
	zᵢ₋₁::NTuple{N,Ty}
	temp::NTuple{N,Ty}
	temp₂::NTuple{N,Ty}
	temp₃::NTuple{N,Tx}
	function BarzilaiBorweinStorage(ρ, state::ADMMState)
		new{length(state.u),typeof(state.u[1]),typeof(state.x)}(
			copy.(state.u), # uᵢ₋₁
			ntuple(i -> ρ[i] * (state.u[i] + state.rᵏ[i]), length(state.u)), # ŷᵢ₋₁
			copy(state.x), # xᵢ₋₁
			copy.(state.z), # zᵢ₋₁
			similar.(state.u), # temp
			similar.(state.u), # temp₂
			ntuple(i -> similar(state.x), length(state.u)), # temp₃
		)
	end
end

"""
	BarzilaiBorweinSpectralPenalty{R,T}

Adaptive ADMM with spectral penalty parameter selection, following the algorithm from
Xu, Figueiredo, and Goldstein (2017). This method uses quasi-Newton estimates to 
adaptively update penalty parameters based on curvature information from the
augmented Lagrangian. This method was inspired by Barzilai-Borwein step size selection
for gradient descent, but adapted for the ADMM framework.

# Arguments
- `rho::R`: Initial penalty parameters (one per regularizer block)
- `eps_cor::T=0.5`: Correlation threshold to determine if curvature can be estimated
- `adp_freq::Int=1`: Frequency of adaptation (every adp_freq iterations)
- `adp_start_iter::Int=2`: Iteration to start adaptation
- `adp_end_iter::Int=typemax(Int)`: Iteration to end adaptation
- `current_iter::Int=0`: Current iteration counter

# References
1. Xu, Z., Figueiredo, M. A., & Goldstein, T. (2017). "Adaptive ADMM with spectral 
   penalty parameter selection." AISTATS.
2. Z. Xu. (2019.) "Alternating optimization: Constrained problems, adversarial networks,
   and robust models." Ph.D. dissertation, University of Maryland, College Park.
3. Lozenski, L., McCann, M. T., & Wohlberg, B. (2025). "An Adaptive Multiparameter
   Penalty Selection Method for Multiconstraint and Multiblock ADMM (No. arXiv:2502.21202).
   arXiv. https://doi.org/10.48550/arXiv.2502.21202
"""
@kwdef mutable struct BarzilaiBorweinSpectralPenalty{R,T} <: PenaltySequence
	rho::R = nothing
	eps_cor::T = 0.5
	adp_freq::Int = 1
	adp_start_iter::Int = 2
	adp_end_iter::Int = typemax(Int)
	current_iter::Int = 0
	storage::Union{Nothing,BarzilaiBorweinStorage} = nothing
	function BarzilaiBorweinSpectralPenalty{R,T}(
		rho::R,
		eps_cor::T,
		adp_freq::Int,
		adp_start_iter::Int,
		adp_end_iter::Int,
		current_iter::Int,
		storage::Union{Nothing,BarzilaiBorweinStorage},
	) where {R,T}
		@assert adp_start_iter >= 2
		@assert adp_start_iter <= adp_end_iter
		@assert adp_freq > 0
		@assert current_iter >= 0
		new{R,T}(
			isnothing(rho) ? nothing : copy(rho),
			eps_cor,
			adp_freq,
			adp_start_iter,
			adp_end_iter,
			current_iter,
			storage,
		)
	end
end

# Constructors
function BarzilaiBorweinSpectralPenalty(rho::R, eps_cor::T, args...) where {R,T}
	BarzilaiBorweinSpectralPenalty{R,T}(rho, eps_cor, args...)
end
function BarzilaiBorweinSpectralPenalty(rho::Union{AbstractVector,Number}; kwargs...)
	BarzilaiBorweinSpectralPenalty(; rho=rho, kwargs...)
end

function reinstantiate_penalty_sequence(
	seq::BarzilaiBorweinSpectralPenalty, ::Type{R}, rho
) where {R}
	final_rho = ensure_correct_value(seq.rho, R, rho)
	BarzilaiBorweinSpectralPenalty{typeof(final_rho),R}(;
		rho=final_rho,
		eps_cor=R(seq.eps_cor),
		adp_freq=seq.adp_freq,
		adp_start_iter=seq.adp_start_iter,
		adp_end_iter=seq.adp_end_iter,
		current_iter=0,
		storage=nothing,
	)
end

function get_next_rho!(
	seq::BarzilaiBorweinSpectralPenalty, iter::ADMMIteration, state::ADMMState
)
	seq.current_iter += 1

	# Initialize storage _after_ first iteration
	if seq.current_iter == max(2, seq.adp_start_iter - seq.adp_freq)
		seq.storage = BarzilaiBorweinStorage(seq.rho, state)
		return seq.rho, false
		# We estimate the penalty parameters only after initialization is done
		# and after adp_start_iter < current_iter < adp_end_iter, and only every adp_freq iterations
	elseif 2 < seq.current_iter && check_iter(seq)
		changed = false
		for i in eachindex(iter.g)
			# Current penalty parameter for this block
			ρ = seq.rho[i]

			# Spectral step size estimation using finite differences
			st = seq.storage
			Δz = @. st.temp[i] = state.z[i] - st.zᵢ₋₁[i]
			Δy = @. st.temp₂[i] = ρ * (state.u[i] - st.uᵢ₋₁[i])
			Δy² = real(dot(Δy, Δy))
			Δz_Δy = real(dot(Δz, Δy))
			Δz² = real(dot(Δz, Δz))
			ŷᵢ = @. st.temp[i] = ρ * (state.u[i] + state.rᵏ[i]) # we already calculated Bᵢ * x - zᵢ₊₁ = rᵏ
			Δŷ = @. st.temp₂[i] = ŷᵢ - st.ŷᵢ₋₁[i]
			BΔx = iter.B[i] * (state.x - st.xᵢ₋₁)
			Δŷ² = real(dot(Δŷ, Δŷ))
			BΔx_Δŷ = real(dot(BΔx, Δŷ))
			BΔx² = real(dot(BΔx, BΔx))

			ϵ = eps(typeof(ρ)) # numerical stability threshold

			α̂ˢᵈ = Δŷ² / (BΔx_Δŷ + ϵ) # sd stands for steepest descent
			α̂ᵐᵍ = BΔx_Δŷ / (BΔx² + ϵ) # mg stands for minimum gradient
			α̂ = curv_adaptive_BB(α̂ˢᵈ, α̂ᵐᵍ)
			β̂ˢᵈ = Δy² / (Δz_Δy + ϵ)
			β̂ᵐᵍ = Δz_Δy / (Δz² + ϵ)
			β̂ = curv_adaptive_BB(β̂ˢᵈ, β̂ᵐᵍ)

			# Safeguarding by assessing the quality of the curvature estimates
			# To enhance numerical stability, the per-theory implementation is commented out
			# and replaced with a more robust check
			ϵᶜᵒʳ = seq.eps_cor
			# αᶜᵒʳ = BΔx_Δŷ / (BΔx² * Δŷ²)
			# βᶜᵒʳ = Δy² / (Δz² * Δy²)
			α̂_is_reliable = BΔx_Δŷ > ϵᶜᵒʳ * (BΔx² * Δŷ²) && abs(α̂) > ϵ && isfinite(α̂) # αᶜᵒʳ > ϵᶜᵒʳ
			β̂_is_reliable = Δy² > ϵᶜᵒʳ * (Δz² * Δy²) && abs(β̂) > ϵ && isfinite(β̂) # βᶜᵒʳ > ϵᶜᵒʳ
			if α̂_is_reliable && β̂_is_reliable
				ρ = sqrt(α̂ * β̂)
			elseif α̂_is_reliable
				ρ = α̂
			elseif β̂_is_reliable
				ρ = β̂
			end
			# If neither curvature can be estimated, keep current ρ

			if α̂_is_reliable || β̂_is_reliable # If ρ is changed
                n = seq.current_iter # - seq.adp_start_iter) ÷ seq.adp_freq
				η = 100
                ω = 2^(-n / η)
				ρⁿᵉʷ = ρ
				ρᵒˡᵈ = seq.rho[i]
                ρ = (1 - ω) * ρᵒˡᵈ + ω * ρⁿᵉʷ
				state.u[i] .*= seq.rho[i] / ρ
				seq.rho[i] = ρ
				changed = true
			end

			# Update storage for next iteration
			st.uᵢ₋₁[i] .= state.u[i]
			st.ŷᵢ₋₁[i] .= ŷᵢ
			st.xᵢ₋₁ .= state.x[i]
			st.zᵢ₋₁[i] .= state.z[i]
		end
		return seq.rho, changed
	end

	return seq.rho, false
end

"""
	curv_adaptive_BB(alpha_num::R, alpha_den::R) where {R}

Hybrid stepsize rule proposed by Zhou et al. (2006), by the
superlinear behavior of the Barzilai-Borwein (BB) method.

References:
1. J.Barzilai and J.M.Borwein, "Two-point step size gradient methods,"
   IMA J. Numer. Anal., vol. 8, pp. 141–148, 1988.
2. B. Zhou, L. Gao, and Y.-H. Dai. Gradient methods with
   adaptive step-sizes. Computational Optimization and Applications,
   35:69–86, 2006.
"""
function curv_adaptive_BB(steepest_descent::R, minimum_gradient::R) where {R}
	ratio = minimum_gradient / steepest_descent
	if ratio > 0.5
		return minimum_gradient
	else
		return steepest_descent - 0.5 * minimum_gradient
	end
end
