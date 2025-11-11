using Test
using ProximalAlgorithms
using LinearAlgebra

# Import internal types for testing - these may be internal API
import ProximalAlgorithms:
    FixedPenalty,
    ResidualBalancingPenalty,
    SpectralRadiusBoundPenalty,
    SpectralRadiusApproximationPenalty,
    reinstantiate_penalty_sequence,
    get_next_rho!,
    ADMMState,
    ADMMIteration,
    CGState

@testset "Penalty Sequences for ADMM" begin

    # Helper function to create a mock ADMMState for testing
    function create_mock_admm_state(rᵏ_norm, sᵏ_norm; z=nothing, z_old=nothing)
        n = length(rᵏ_norm)
        R = eltype(sᵏ_norm)

        # Create mock arrays if not provided
        if z === nothing
            z = [randn(R, 10) for _ in 1:n]
        end
        if z_old === nothing
            z_old = [randn(R, 10) for _ in 1:n]
        end

        # Create a proper ADMMState
        x = randn(R, 10)  # primal variable
        u = [randn(R, 10) for _ in 1:n]  # dual variables
        sᵏ = ntuple(_ -> similar(x), n)
        tempˣ = ntuple(_ -> similar(x), n)
        rᵏ = similar.(u)
        Bx = similar.(u)
        R = real(eltype(x))
        Δx_norm = zero(R)
        ϵᵖʳⁱ = ones(R, n)
        ϵᵈᵘᵃ = ones(R, n)
        cg_operator = LinearAlgebra.I  # Simple identity operator
        cg_state = CGState(x, similar(x))

        state = ADMMState(
            x,
            u,
            z,
            z_old,
            rᵏ,
            sᵏ,
            tempˣ,
            Bx,
            Δx_norm,
            rᵏ_norm,
            sᵏ_norm,
            ϵᵖʳⁱ,
            ϵᵈᵘᵃ,
            cg_state,
            cg_operator,
            0,
        )

        return state
    end

    # Helper function to create a mock ADMMIteration for testing
    function create_mock_admm_iteration(n_reg=2)
        # Create mock functions and operators
        g = ntuple(i -> x -> 0.5 * norm(x)^2, n_reg)  # Simple quadratic functions
        B = ntuple(i -> LinearAlgebra.I, n_reg)       # Identity operators

        # Create a minimal ADMMIteration with the required fields
        x0 = randn(10)
        b = randn(10)
        R = Float64

        penalty_seq = FixedPenalty(ones(n_reg))  # Create with right number of elements

        # Create the iteration directly with struct constructor
        ADMMIteration(
            x0,          # x0
            nothing,     # A  
            nothing,     # b
            nothing,     # AHb
            g,          # g
            B,          # B
            nothing,     # P
            false,       # P_is_inverse
            R(1e-6),     # cg_tol
            100,         # cg_maxit
            nothing,     # y0
            nothing,     # z0
            penalty_seq,  # penalty_sequence
        )
    end

    @testset "FixedPenalty" begin
        rho_init = [1.0, 2.0, 3.0]
        seq = FixedPenalty(rho_init)
        iter = create_mock_admm_iteration(3)

        # Test that penalties remain fixed
        state = create_mock_admm_state([0.1, 0.2, 0.3], [0.05, 0.1, 0.15])
        rho_new, changed = get_next_rho!(seq, iter, state)

        @test rho_new == rho_init
        @test !changed  # FixedPenalty should never change
        @test seq.rho == rho_init  # Original should be unchanged

        # Test with different residuals - should still be fixed
        state2 = create_mock_admm_state([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        rho_new2, changed2 = get_next_rho!(seq, iter, state2)
        @test rho_new2 == rho_init
        @test !changed2
    end

    @testset "ResidualBalancingPenalty" begin
        rho_init = [1.0, 2.0]
        mu = 10.0
        tau = 2.0
        seq = ResidualBalancingPenalty(rho_init, mu=mu, tau=tau)
        iter = create_mock_admm_iteration(2)

        @test seq.mu == mu
        @test seq.tau == tau
        @test seq.rho == rho_init

        # Test case 1: primal > mu * dual (should increase rho)
        state = create_mock_admm_state([1.0, 1.0], [0.05, 0.05])  # primal/dual = 20 > mu=10
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init)
        @test !changed  # Should not change on first call
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init .* tau)
        @test changed

        # Reset penalty
        seq = ResidualBalancingPenalty(rho_init, mu=mu, tau=tau)

        # Test case 2: dual > mu * primal (should decrease rho)
        state = create_mock_admm_state([0.05, 0.05], [1.0, 1.0])  # dual/primal = 20 > mu=10
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init)
        @test !changed  # Should not change on first call
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init ./ tau)
        @test changed

        # Test case 3: balanced residuals (should not change)
        seq = ResidualBalancingPenalty(rho_init, mu=mu, tau=tau)
        state = create_mock_admm_state([1.0, 1.0], [0.5, 0.5])  # ratio = 2 < mu=10
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init)
        @test !changed
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init)
        @test !changed

        # Test constructor with positional rho argument
        seq_pos = ResidualBalancingPenalty(rho_init, mu=mu, tau=tau)
        @test seq_pos.rho == rho_init

        # Test constructor with scalar rho
        seq_scalar = ResidualBalancingPenalty(1.5, mu=mu, tau=tau)
        @test seq_scalar.rho == 1.5
    end

    #=@testset "WohlbergPenalty" begin
        rho_init = [1.0, 2.0]
        mu = 10.0
        tau_init = [2.0, 2.0]
        tau_max = 10.0

        seq = WohlbergPenalty(rho=rho_init, mu=mu, tau=tau_init, tau_max=tau_max)
        iter = create_mock_admm_iteration(2)

        @test seq.mu == mu
        @test seq.tau == tau_init  # Per-block tau initialization
        @test seq.tau_max == tau_max

        # Test with residuals where primal dominates in first block, dual in second
        state = create_mock_admm_state([20.0, 1.0], [1.0, 20.0])  # ratios: 20.0, 0.05
        initial_tau = copy(seq.tau)
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test initial_tau == seq.tau  # tau should not change on first call
        @test !changed

        rho_new, changed = get_next_rho!(seq, iter, state)

        # First block: primal >> dual (ratio = 20 > mu = 10)
        # Should increase tau[1] and then multiply rho[1] by tau[1]
        @test seq.tau[1] > initial_tau[1]  # tau adapted upward
        @test rho_new[1] > rho_init[1]     # rho increased

        # Second block: dual >> primal (ratio = 0.05 < 1/mu = 0.1)  
        # Should increase tau[2] and then divide rho[2] by tau[2]
        @test seq.tau[2] > initial_tau[2]  # tau adapted upward
        @test rho_new[2] < rho_init[2]     # rho decreased

        @test changed  # WohlbergPenalty changes when residuals are imbalanced

        # Test with balanced residuals (no change expected)
        seq2 = WohlbergPenalty([1.0, 2.0]; mu=10.0, tau=tau_init)
        state_balanced = create_mock_admm_state([1.0, 2.0], [1.0, 2.0])  # ratios: 1.0, 1.0
        rho_new2, changed2 = get_next_rho!(seq2, iter, state_balanced)
        rho_new2, changed2 = get_next_rho!(seq2, iter, state_balanced)

        @test seq2.tau == [1.0, 1.0]
        @test rho_new2 == [1.0, 2.0]    # rho should not change for balanced residuals
        @test !changed2  # No change when residuals are balanced
    end

    @testset " BarzilaiBorweinSpectralPenalty" begin
        rho_init = [1.0, 2.0]
        seq =  BarzilaiBorweinSpectralPenalty(rho=rho_init)
        iter = create_mock_admm_iteration(2)

        @test seq.rho == rho_init
        @test seq.current_iter == 0

        # Test first iteration (should return unchanged rho)
        state = create_mock_admm_state([0.8, 0.8], [0.05, 0.05])
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test rho_new == rho_init
        @test !changed
        @test seq.current_iter == 1

        # Test second iteration (initializes storage)
        state2 = create_mock_admm_state([0.7, 0.7], [0.04, 0.04])
        rho_new2, changed2 = get_next_rho!(seq, iter, state2)
        @test seq.current_iter == 2
        # Storage should be initialized but no adaptation yet
        @test !changed2
    end=#

    @testset "Type Consistency" begin
        # Test that reinstantiate_penalty_sequence works correctly for all penalty types
        for T in [Float32, Float64]
            rho = T[1, 2]

            # Test FixedPenalty
            seq1 = FixedPenalty(rho)
            seq1_converted = reinstantiate_penalty_sequence(seq1, T, nothing)
            @test eltype(seq1_converted.rho) == T

            # Test ResidualBalancingPenalty
            seq2 = ResidualBalancingPenalty(rho=rho, mu=T(10), tau=T(2))
            seq2_converted = reinstantiate_penalty_sequence(seq2, T, nothing)
            @test eltype(seq2_converted.rho) == T
            @test typeof(seq2_converted.mu) == T
            @test typeof(seq2_converted.tau) == T

            #= Test WohlbergPenalty
            seq3 = WohlbergPenalty(rho=rho, mu=T(10), tau=fill(T(2), length(rho)), tau_max=T(10))
            seq3_converted = reinstantiate_penalty_sequence(seq3, T, nothing)
            @test eltype(seq3_converted.rho) == T
            @test typeof(seq3_converted.mu) == T
            @test eltype(seq3_converted.tau) == T
            @test typeof(seq3_converted.tau_max) == T

            # Test  BarzilaiBorweinSpectralPenalty
            seq4 =  BarzilaiBorweinSpectralPenalty(rho=rho)
            seq4_converted = reinstantiate_penalty_sequence(seq4, T, nothing)
            @test eltype(seq4_converted.rho) == T
            @test eltype(seq4_converted.orthval) == T
            @test eltype(seq4_converted.minval) == T=#
        end
    end

    @testset "Edge Cases" begin
        rho_init = [1.0, 2.0]
        iter = create_mock_admm_iteration(2)

        # Test with zero residuals
        seq = ResidualBalancingPenalty(rho_init)
        state = create_mock_admm_state([0.0, 0.0], [0.0, 0.0])
        rho_new, changed = get_next_rho!(seq, iter, state)
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .≈ rho_init)  # Should not change with zero residuals
        @test !changed

        # Test with very large residuals
        seq = ResidualBalancingPenalty([1.0, 2.0])
        state = create_mock_admm_state([1e10, 1e10], [1e-10, 1e-10])
        rho_new, changed = get_next_rho!(seq, iter, state)
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test all(rho_new .> rho_init)  # Should increase
        @test changed

        # Test with single element
        rho_single = [1.0]
        seq = ResidualBalancingPenalty(rho_single)
        iter_single = create_mock_admm_iteration(1)
        state = create_mock_admm_state([1.0], [0.05])  # ratio = 20 > mu=10
        rho_new, changed = get_next_rho!(seq, iter_single, state)
        rho_new, changed = get_next_rho!(seq, iter_single, state)
        @test length(rho_new) == 1
        @test rho_new[1] > rho_single[1]
        @test changed
    end

    @testset "Constructor Variants" begin
        # Test different constructor patterns

        # FixedPenalty
        @test FixedPenalty([1.0, 2.0]).rho == [1.0, 2.0]
        @test FixedPenalty(1.5).rho == 1.5
        @test FixedPenalty().rho === nothing

        # ResidualBalancingPenalty
        @test ResidualBalancingPenalty().rho === nothing
        @test ResidualBalancingPenalty([1.0, 2.0]).rho == [1.0, 2.0]
        @test ResidualBalancingPenalty(1.5).rho == 1.5

        #= WohlbergPenalty
        @test WohlbergPenalty().rho === nothing
        @test WohlbergPenalty([1.0, 2.0]).rho == [1.0, 2.0]
        @test WohlbergPenalty(1.5).rho == 1.5

        # Test that tau is initialized per-block
        seq_multi = WohlbergPenalty([1.0, 2.0, 3.0])
        seq_multi = reinstantiate_penalty_sequence(seq_multi, Float64, nothing)
        @test length(seq_multi.tau) == 3
        @test all(seq_multi.tau .== 2.0)  # default tau_init

        #  BarzilaiBorweinSpectralPenalty
        @test  BarzilaiBorweinSpectralPenalty().rho === nothing
        @test  BarzilaiBorweinSpectralPenalty([1.0, 2.0]).rho == [1.0, 2.0]
        @test  BarzilaiBorweinSpectralPenalty(1.5).rho == 1.5=#
    end

    @testset "Convert Types with Provided Rho" begin
        # Test that reinstantiate_penalty_sequence can override rho
        original_seq = FixedPenalty([1.0, 2.0])
        new_rho = Float32[3.0, 4.0, 5.0]

        converted_seq = reinstantiate_penalty_sequence(original_seq, Float32, new_rho)
        @test converted_seq.rho == new_rho
        @test eltype(converted_seq.rho) == Float32
        @test length(converted_seq.rho) == 3  # Different length than original

        # Test with ResidualBalancingPenalty
        original_seq2 = ResidualBalancingPenalty(rho=[1.0, 2.0], mu=10.0, tau=2.0)
        converted_seq2 = reinstantiate_penalty_sequence(original_seq2, Float32, new_rho)
        @test converted_seq2.rho == new_rho
        @test typeof(converted_seq2.mu) == Float32
        @test typeof(converted_seq2.tau) == Float32

        #= Test with WohlbergPenalty 
        original_seq3 = WohlbergPenalty(rho=[1.0, 2.0], mu=10.0, tau=[2.0, 2.0])
        converted_seq3 = reinstantiate_penalty_sequence(original_seq3, Float32, new_rho)
        @test converted_seq3.rho == new_rho
        @test typeof(converted_seq3.mu) == Float32
        @test eltype(converted_seq3.tau) == Float32
        @test length(converted_seq3.tau) == length(new_rho)  # tau should match new rho length=#
    end

    @testset "SpectralRadiusBoundPenalty" begin
        rho_init = [1.0, 2.0]
        seq = SpectralRadiusBoundPenalty(rho=rho_init)
        seq = reinstantiate_penalty_sequence(seq, Float64, rho_init)
        iter = create_mock_admm_iteration(2)

        @test seq.rho == rho_init
        @test seq.current_iter == 0

        # Test first iteration (should return unchanged rho)
        state = create_mock_admm_state([0.8, 0.8], [0.05, 0.05])
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test rho_new == rho_init
        @test !changed
        @test seq.current_iter == 1

        # Test second iteration (initializes storage)
        state2 = create_mock_admm_state([0.7, 0.7], [0.04, 0.04])
        rho_new2, changed2 = get_next_rho!(seq, iter, state2)
        @test seq.current_iter == 2
        @test !changed2

        # Test adaptation logic
        state3 = create_mock_admm_state([1.0, 1.0], [0.5, 0.5])
        rho_new3, changed3 = get_next_rho!(seq, iter, state3)
        @test changed3
        @test all(rho_new3 .!= rho_init)
    end

    @testset "SpectralRadiusApproximationPenalty" begin
        rho_init = [1.0, 2.0]
        seq = SpectralRadiusApproximationPenalty(rho=rho_init)
        seq = reinstantiate_penalty_sequence(seq, Float64, rho_init)
        iter = create_mock_admm_iteration(2)

        @test seq.rho == rho_init
        @test seq.current_iter == 0

        # Test first iteration (should return unchanged rho)
        state = create_mock_admm_state([0.8, 0.8], [0.05, 0.05])
        rho_new, changed = get_next_rho!(seq, iter, state)
        @test rho_new == rho_init
        @test !changed
        @test seq.current_iter == 1

        # Test second iteration (initializes storage)
        state2 = create_mock_admm_state([0.7, 0.7], [0.04, 0.04])
        rho_new2, changed2 = get_next_rho!(seq, iter, state2)
        @test seq.current_iter == 2
        @test !changed2

        # Test adaptation logic
        state3 = create_mock_admm_state([1.0, 1.0], [0.5, 0.5])
        rho_new3, changed3 = get_next_rho!(seq, iter, state3)
        @test changed3
        @test all(rho_new3 .!= rho_init)
    end
end
