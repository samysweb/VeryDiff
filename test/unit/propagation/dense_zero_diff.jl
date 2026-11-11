using VeryDiff
import VeryDiff: VerificationTask, PropState, prepare_prop_state!, propagate!, GeminiNetwork, Zonotope, zono_bounds, reset_ps!
using Test
using LinearAlgebra
using Random

@testset "Dense Layer Propagation Tests (Zero Diff)" begin
    @info "Starting Dense Layer Propagation Tests (Zero Diff)"
    
    if "VERYDIFF_TEST_SEED" in keys(ENV)
        test_seed = parse(Int, ENV["VERYDIFF_TEST_SEED"])
        @info "Using VERYDIFF_TEST_SEED: $(test_seed)"
    else
        test_seed = rand(1:999999)
    end
    Random.seed!(test_seed)
    @info "Test seed: $(test_seed)"

    execution_difference = 1e-2
    
    @testset "Basic Gemini Network Propagation" begin
        input_dim = 5
        num_layers = rand(5:15)
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(20:50) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create two identical networks with same structure
        N1, N2 = make_dense_pair(input_dim, layer_dims; identical=true)
        
        # Create Gemini Network
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds [-1, 1]^input_dim
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask to encode the property with secondary distances
        verification_task = create_verification_task(low, high; with_secondary=true, secondary_scale=execution_difference)
        
        # Create PropState and initialize with verification task
        prop_state = PropState(true)
        prepare_prop_state!(prop_state, verification_task)
        
        # Propagate through Gemini Network
        prop_state = propagate!(N_gemini, prop_state)
        
        # Extract output zonotopes
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        
        # Basic checks
        @test !isnothing(Zout)
        @test size(Zout.Z₁.c) == (10,)  # Output dimension should be 10
        @test size(Zout.Z₂.c) == (10,)
    end
    
    @testset "Memory Allocation Reduction on Second Run" begin
        input_dim = 5
        num_layers = rand(5:15)
        
        # Create layer dimensions (same for both networks)
        layer_dims = [rand(400:600) for _ in 1:(num_layers-1)]
        push!(layer_dims, 10)  # Output dimension is 10
        
        # Create networks with same structure and identical weights
        N1, N2 = make_dense_pair(input_dim, layer_dims; identical=true)
        N_gemini = GeminiNetwork(N1, N2)
        
        # Create input bounds
        low = fill(-1.0, input_dim)
        high = fill(1.0, input_dim)
        
        # Create VerificationTask (with secondary distances to track zero diff)
        verification_task = create_verification_task(low, high; with_secondary=true, secondary_scale=execution_difference)
        
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
        num_layers = rand(5:15)
        num_samples = 20_000
        tolerance = 1e-4
        
        for depth in 1:15
            # Create layer dimensions (same for both networks)
            layer_dims = [rand(20:100) for _ in 1:depth]
            push!(layer_dims, 10)  # Output dimension is 10
            depth = length(layer_dims)
            
            # Create networks with same structure and identical weights
            N1, N2 = make_dense_pair(input_dim, layer_dims; identical=true)
            
            # Create input bounds [-1, 1]^input_dim
            low = fill(-1.0, input_dim)
            high = fill(1.0, input_dim)

            # Sample points from input space and propagate through both networks
            # Secondary dimensions share same span as primary
            sec_low = fill(-execution_difference, input_dim * 2)
            sec_high = fill(execution_difference, input_dim * 2)
            input_samples = sample_points_in_hypercube(low, high, num_samples; secondary_low=sec_low, secondary_high=sec_high)

            N_gemini = GeminiNetwork(N1, N2)

            N1 = executable_network(N1)
            N2 = executable_network(N2)
        
            # Create VerificationTask with secondary distances to capture zero difference
            verification_task = create_verification_task(low, high; with_secondary=true, secondary_scale=execution_difference)
            
            # Propagate zonotope through network
            prop_state = PropState(true)
            prepare_prop_state!(prop_state, verification_task)
            # Print first Zonotope:
            #@info "Initial Zonotope₁ Generators: $(prop_state.zono_storage.zonotopes[1].zonotope.Z₁.Gs)"
            #@info "Initial Zonotope₂ Generators: $(prop_state.zono_storage.zonotopes[1].zonotope.Z₂.Gs)"
            #@info "Initial Zonotope∂ Generators: $(prop_state.zono_storage.zonotopes[1].zonotope.∂Z.Gs)"
            prop_state = propagate!(N_gemini, prop_state)
            Zout = prop_state.zono_storage.zonotopes[end].zonotope

            # Get output bounds from zonotope
            bounds_z1 = zono_bounds(Zout.Z₁)
            bounds_z2 = zono_bounds(Zout.Z₂)
            bounds_z∂ = zono_bounds(Zout.∂Z)
            # @info "Output bounds for Network 1: $bounds_z1"
            # @info "Output bounds for Network 2: $bounds_z2"
            # @info "Output bounds for Difference Zonotope: $bounds_z∂"
            
            violations = 0
            n1_violations = 0
            n2_violations = 0
            diff_violations = 0
            for i in 1:num_samples
                prim = input_samples[1:input_dim, i]
                x = prim
                sec1 = input_samples[input_dim+1:input_dim*2, i]
                sec2 = input_samples[input_dim*2+1:end, i]
                x1 = x .+ sec1
                x2 = x .+ sec2
                sec1 ./= execution_difference
                sec2 ./= execution_difference

                Zin = prop_state.zono_storage.zonotopes[1].zonotope.Z₁
                @assert Zin.c .+ Zin.Gs[1]*prim .+ Zin.Gs[2]*sec1 ≈ x1 atol=1e-8
                Zin2 = prop_state.zono_storage.zonotopes[1].zonotope.Z₂
                @assert Zin2.c .+ Zin2.Gs[1]*prim .+ Zin2.Gs[2]*sec2 ≈ x2 atol=1e-8
                
                # Propagate through N1
                y1 = N1(x1)
                
                # Propagate through N2
                y2 = N2(x2)

                Zout = prop_state.zono_storage.zonotopes[end].zonotope
                @test Zout.Z₁.c .+ Zout.Z₁.Gs[1]*prim .+ Zout.Z₁.Gs[2]*sec1 ≈ y1 atol=1e-8
                @test Zout.Z₂.c .+ Zout.Z₂.Gs[1]*prim .+ Zout.Z₂.Gs[2]*sec2 ≈ y2 atol=1e-8
                @test Zout.∂Z.c .+ Zout.∂Z.Gs[1]*sec1 .+ Zout.∂Z.Gs[2]*sec2 ≈ (y1 .- y2) atol=1e-8
                
                # Check if outputs are within bounds
                for d in 1:size(y1, 1)
                    if y1[d] < bounds_z1[d, 1] - tolerance || y1[d] > bounds_z1[d, 2] + tolerance
                        n1_violations += 1
                        @info "Difference violation at dimension $d in Net 1: $(y1[d]) not in [$(bounds_z1[d, 1]), $(bounds_z1[d, 2])]"
                        @info "Input: $(x1)"
                        @info "Input Samples: $(input_samples[:, i])"
                        # Output dimension d of Zonotope:
                        @info "Zonotope₁ dimension $d: $(Zout.Z₁.c[d]) ± "
                        for g in Zout.Z₁.Gs
                            @info "Generator: $(g[d, :]) (sum: $(sum(abs.(g[d, :]))))"
                        end
                        @info "Failed"
                        #@info Zout.Z₁
                        return nothing
                    end
                    
                    if y2[d] < bounds_z2[d, 1] - tolerance || y2[d] > bounds_z2[d, 2] + tolerance
                        n2_violations += 1
                        # @info "Difference violation at dimension $d in Net 2: $(y2[d]) not in [$(bounds_z2[d, 1]), $(bounds_z2[d, 2])]"
                        # @info "Input: $(x2)"
                    end
                    if (y1[d] - y2[d]) < bounds_z∂[d, 1] - tolerance || (y1[d] - y2[d]) > bounds_z∂[d, 2] + tolerance
                        diff_violations += 1
                        # @info "Difference violation at dimension $d: $(y1[d] - y2[d]) not in [$(bounds_z∂[d, 1]), $(bounds_z∂[d, 2])]"
                        # @info "Inputs: $(x1), $(x2) (Difference: $(x1 .- x2))"
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
