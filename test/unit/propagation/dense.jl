using VeryDiff
import VeryDiff: VerificationTask, PropState, prepare_prop_state!, propagate!, GeminiNetwork, Zonotope, zono_bounds, reset_ps!
using Test
using LinearAlgebra
using Random

@testset "Dense Layer Propagation Tests" begin
    @info "Starting Dense Layer Propagation Tests"
    
    if "VERYDIFF_TEST_SEED" in keys(ENV)
        test_seed = parse(Int, ENV["VERYDIFF_TEST_SEED"])
        @info "Using VERYDIFF_TEST_SEED: $(test_seed)"
    else
        test_seed = rand(1:999999)
    end
    Random.seed!(test_seed)
    @info "Test seed: $(test_seed)"
    
    @testset "Basic Gemini Network Propagation" begin
        input_dim = 5
        num_layers = rand(5:15)
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(20:50) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create two randomized networks with same structure
        N1, N2 = make_dense_pair(input_dim, layer_dims)
        
        # Create Gemini Network
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds [-1, 1]^input_dim
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask to encode the property
        verification_task = create_verification_task(low, high)
        
        # Create PropState and initialize with verification task
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        
        # Propagate through Gemini Network
        prop_state = propagate!(N_gemini, prop_state)
        
        # Extract output zonotopes
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        
        # Basic checks
        # @test !isnothing(Zout)
        # @test size(Zout.Z₁.c) == (10,)  # Output dimension should be 10
        # @test size(Zout.Z₂.c) == (10,)
    end
    
    @testset "Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        num_layers = rand(5:15)
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(400:600) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create networks with same structure
        N1, N2 = make_dense_pair(input_dim, layer_dims)
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask
        verification_task = create_verification_task(low, high)
        
        # First propagation - counts allocations
        prop_state_1 = PropState(true)
        prepare_prop_state!(prop_state_1, verification_task)
        res_1, time_1, alloc_1, _ = @timed propagate!(N_gemini, prop_state_1)
        
        # Second propagation - should allocate less
        reset_ps!(prop_state_1)
        prepare_prop_state!(prop_state_1, verification_task)
        res_2, time_2, alloc_2, _ = @timed propagate!(N_gemini, prop_state_1)
        
        @info "First run allocations: $alloc_1 bytes, Second run allocations: $alloc_2 bytes"
        @info "First run time: $time_1 s, Second run time: $time_2 s"
        
        # Check that second run allocates less (allowing for small variance)
        @test alloc_2 < alloc_1 * 0.4
    end
    
    @testset "Sampled Points Within Output Bounds" begin
        input_dim = 10
        num_samples = 20_000
        tolerance = 1e-4

        for depth in 1:15
        
            # Create layer dimensions (same for both networks)
            layer_dims = [rand(20:100) for _ in 1:depth]
            push!(layer_dims, 10)  # Output dimension is 10
            
            @debug "Creating networks..."
            # Create networks with same structure
            N1, N2 = make_dense_pair(input_dim, layer_dims)
            depth = length(layer_dims)

            # Create input bounds [-1, 1]^input_dim
            low = fill(-1.0, input_dim)
            high = fill(1.0, input_dim)
            # Sample points from input space and propagate through both networks
            input_samples = sample_points_in_hypercube(low, high, num_samples)

            
            N_gemini = GeminiNetwork(N1, N2)

            N1 = executable_network(N1)
            N2 = executable_network(N2)
            
            # Create VerificationTask
            verification_task = create_verification_task(low, high)
            
            # Propagate zonotope through network
            prop_state = PropState(true)
            prepare_prop_state!(prop_state, verification_task)
            prop_state = propagate!(N_gemini, prop_state)
            @debug "Propagation through Gemini Network complete."
            @debug "$(length(prop_state.zono_storage.zonotopes)) zonotopes in storage."
            Zout = prop_state.zono_storage.zonotopes[end].zonotope
            
            # Get output bounds from zonotope
            bounds_z1 = zono_bounds(Zout.Z₁)
            bounds_z2 = zono_bounds(Zout.Z₂)
            bounds_z∂ = zono_bounds(Zout.∂Z)
            # @info "Output bounds for Network 1: $bounds_z1"
            # @info "Output bounds for Network 2: $bounds_z2"
            
            n1_violations = 0
            n2_violations = 0
            diff_violations = 0
            for i in 1:num_samples
                x = input_samples[:, i]

                Zin = prop_state.zono_storage.zonotopes[1].zonotope.Z₁
                @assert Zin.c .+ Zin.Gs[1]*x ≈ x atol=1e-8
                Zin2 = prop_state.zono_storage.zonotopes[1].zonotope.Z₂
                @assert Zin2.c .+ Zin2.Gs[1]*x ≈ x atol=1e-8
                
                # Propagate through N1
                y1 = N1(x)
                
                # Propagate through N2
                y2 = N2(x)

                Zout = prop_state.zono_storage.zonotopes[end].zonotope
                @test Zout.Z₁.c .+ Zout.Z₁.Gs[1]*x ≈ y1 atol=1e-8
                @test Zout.Z₂.c .+ Zout.Z₂.Gs[1]*x ≈ y2 atol=1e-8
                if length(Zout.∂Z.Gs) >= 1
                    @test Zout.∂Z.c .+ Zout.∂Z.Gs[1]*x ≈ (y1 .- y2) atol=1e-8
                else
                    @test Zout.∂Z.c ≈ (y1 .- y2) atol=1e-8
                    if !isapprox(Zout.∂Z.c, (y1 .- y2); atol=1e-8)
                        @info "Diff output mismatch: Zonotope diff $(Zout.∂Z.c) vs actual diff $(y1 .- y2)"
                        @info "Input x: $x"
                        @info "Zonotope Z1 c: $(Zout.Z₁.c), Gs: $(Zout.Z₁.Gs)"
                        @info "Zonotope Z2 c: $(Zout.Z₂.c), Gs: $(Zout.Z₂.Gs)"
                        return
                    end
                end
                
                # Check if outputs are within bounds
                for d in 1:size(y1, 1)
                    if y1[d] < bounds_z1[d, 1] - tolerance || y1[d] > bounds_z1[d, 2] + tolerance
                        n1_violations += 1
                        @info "Difference violation at dimension $d in Net 1: $(y1[d]) not in [$(bounds_z1[d, 1]), $(bounds_z1[d, 2])]"
                    end
                    
                    if y2[d] < bounds_z2[d, 1] - tolerance || y2[d] > bounds_z2[d, 2] + tolerance
                        n2_violations += 1
                        @info "Difference violation at dimension $d in Net 2: $(y2[d]) not in [$(bounds_z2[d, 1]), $(bounds_z2[d, 2])]"
                    end
                    if (y1[d] - y2[d]) < bounds_z∂[d, 1] - tolerance || (y1[d] - y2[d]) > bounds_z∂[d, 2] + tolerance
                        diff_violations += 1
                        @info "Difference violation at dimension $d: $(y1[d] - y2[d]) not in [$(bounds_z∂[d, 1]), $(bounds_z∂[d, 2])]"
                    end
                end
            end
            violations = n1_violations + n2_violations + diff_violations
            @info "Found $violations bound violations out of $(num_samples * 30) total checks"
            if violations > 0
                @info "Of which N1: $n1_violations"
                @info "Of which N2: $n2_violations"
                @info "Of which Diff: $diff_violations"
            end
            # All sampled points should be within bounds
            @test violations == 0
        end
    end
    
end
