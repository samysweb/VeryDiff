"""
Comparison fuzzer that runs Julia and PyTorch implementations in parallel
and verifies they produce matching results layer-by-layer.
"""

using VeryDiff
import VeryDiff: VerificationTask, PropState, prepare_prop_state!, propagate!, GeminiNetwork, Zonotope, DiffZonotope, zono_bounds, reset_ps!, get_layers
using Test
using LinearAlgebra
using Random
using PyCall
import VNNLib: NNLoader

# Include utility functions
include("unit/propagation/utils.jl")

# Import PyTorch implementation
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "..", "torch"))
py_src = pyimport("src")

"""
    julia_to_python_network(network::VeryDiff.Network)

Convert Julia network to PyTorch network.
"""
function julia_to_python_network(network::VeryDiff.Network)
    torch = pyimport("torch")
    nn = pyimport("torch.nn")
    
    py_layers = PyObject[]
    for layer in network.layers
        if isa(layer, NNLoader.Dense)
            # Create PyTorch Linear layer
            in_features = size(layer.W, 2)
            out_features = size(layer.W, 1)
            py_layer = nn.Linear(in_features, out_features)
            
            # Copy weights (note: PyTorch uses transposed convention)
            py_layer.weight.data = torch.from_numpy(collect(layer.W))
            py_layer.bias.data = torch.from_numpy(collect(layer.b))
            
            push!(py_layers, py_layer)
        elseif isa(layer, NNLoader.ReLU)
            push!(py_layers, nn.ReLU())
        else
            error("Unsupported layer type: $(typeof(layer))")
        end
    end
    
    return nn.Sequential(py_layers...)
end

"""
    julia_zonotope_to_python(Z::Zonotope)

Convert Julia Zonotope to Python Zonotope.
"""
function julia_zonotope_to_python(Z::Zonotope)
    torch = pyimport("torch")
    
    # Convert generator matrices
    py_Gs = [torch.from_numpy(collect(G)) for G in Z.Gs]
    
    # Convert center
    py_c = torch.from_numpy(collect(Z.c))
    
    # Convert generator IDs
    py_generator_ids = py_src.SortedVector(collect(Z.generator_ids.data))
    
    # Convert owned generators
    py_owned = isnothing(Z.owned_generators) ? nothing : (Z.owned_generators - 1)  # Python 0-indexed
    
    return py_src.Zonotope(py_Gs, py_c, py_generator_ids, py_owned)
end

"""
    julia_diffzonotope_to_python(DZ::DiffZonotope)

Convert Julia DiffZonotope to Python DiffZonotope.
"""
function julia_diffzonotope_to_python(DZ::DiffZonotope)
    py_Z1 = julia_zonotope_to_python(DZ.Z₁)
    py_Z2 = julia_zonotope_to_python(DZ.Z₂)
    py_dZ = julia_zonotope_to_python(DZ.∂Z)
    
    return py_src.DiffZonotope(py_Z1, py_Z2, py_dZ)
end

"""
    python_zonotope_to_julia(py_Z)

Convert Python Zonotope to Julia Zonotope.
"""
function python_zonotope_to_julia(py_Z)
    # Convert generator matrices
    Gs = AbstractMatrix{Float64}[py_Z.Gs[i].detach().numpy() for i in 1:(length(py_Z.Gs))]
    
    # Convert center
    c = py_Z.c.detach().numpy()
    
    # Convert generator IDs
    generator_ids = VeryDiff.Definitions.SortedVector{Int64}(collect(py_Z.generator_ids.data))
    
    # Convert owned generators (Python is 0-indexed, Julia is 1-indexed)
    owned = isnothing(py_Z.owned_generators) ? nothing : (py_Z.owned_generators + 1)
    
    return Zonotope(Gs, c, nothing, generator_ids, owned)
end

"""
    python_diffzonotope_to_julia(py_DZ)

Convert Python DiffZonotope to Julia DiffZonotope.
"""
function python_diffzonotope_to_julia(py_DZ)
    Z1 = python_zonotope_to_julia(py_DZ.Z1)
    Z2 = python_zonotope_to_julia(py_DZ.Z2)
    dZ = python_zonotope_to_julia(py_DZ.dZ)
    
    return DiffZonotope(Z1, Z2, dZ)
end

"""
    compare_zonotopes(Z_julia::Zonotope, Z_python, label::String; rtol=1e-5, atol=1e-6)

Compare Julia and Python zonotopes and assert they match.
"""
function compare_zonotopes(Z_julia::Zonotope, Z_python, label::String; rtol=1e-5, atol=1e-6)
    # Convert Python zonotope to Julia for comparison
    Z_py_julia = python_zonotope_to_julia(Z_python)
    
    # Compare centers
    @test isapprox(Z_julia.c, Z_py_julia.c, rtol=rtol, atol=atol)
    if !(isapprox(Z_julia.c, Z_py_julia.c, rtol=rtol, atol=atol)) 
        @warn "$label: Centers differ" Z_julia.c Z_py_julia.c
    end
    
    # Compare number of generators
    @test length(Z_julia.Gs) == length(Z_py_julia.Gs) 
    if length(Z_julia.Gs) != length(Z_py_julia.Gs)
        @warn "$label: Different number of generator matrices" length(Z_julia.Gs) length(Z_py_julia.Gs)
    end
    
    # Compare generator matrices
    for (i, (G_jl, G_py)) in enumerate(zip(Z_julia.Gs, Z_py_julia.Gs))
        @test isapprox(G_jl, G_py, rtol=rtol, atol=atol)
        if !(isapprox(G_jl, G_py, rtol=rtol, atol=atol))
            @warn "$label: Generator $i differs" maximum(abs.(G_jl .- G_py))
        end
    end
    
    # Compare generator IDs
    @test Z_julia.generator_ids.data == Z_py_julia.generator_ids.data 
    if Z_julia.generator_ids.data != Z_py_julia.generator_ids.data
        @warn "$label: Generator IDs differ" Z_julia.generator_ids.data Z_py_julia.generator_ids.data
    end 
end

"""
    compare_diffzonotopes(DZ_julia::DiffZonotope, DZ_python, label::String; rtol=1e-5, atol=1e-6)

Compare Julia and Python differential zonotopes.
"""
function compare_diffzonotopes(DZ_julia::DiffZonotope, DZ_python, label::String; rtol=1e-5, atol=1e-6)
    compare_zonotopes(DZ_julia.Z₁, DZ_python.Z1, "$label Z₁", rtol=rtol, atol=atol)
    compare_zonotopes(DZ_julia.Z₂, DZ_python.Z2, "$label Z₂", rtol=rtol, atol=atol)
    compare_zonotopes(DZ_julia.∂Z, DZ_python.dZ, "$label ∂Z", rtol=rtol, atol=atol)
end

"""
    test_layer_by_layer_comparison(;input_dim=5, num_layers=5, use_relu=true, rtol=1e-5, atol=1e-6)

Test that Julia and Python implementations produce matching results layer-by-layer.
"""
function test_layer_by_layer_comparison(;input_dim=5, num_layers=5, use_relu=true, rtol=1e-5, atol=1e-6)
    # Create layer dimensions
    layer_dims = [rand(10:30) for _ in 1:(num_layers-1)]
    push!(layer_dims, 10)  # Output dimension
    
    @info "Testing network: input_dim=$input_dim, layers=$layer_dims, relu=$use_relu"
    
    # Create Julia networks
    N1_jl, N2_jl = make_dense_pair(input_dim, layer_dims; relu=use_relu)
    N_gemini_jl = GeminiNetwork(N1_jl, N2_jl)
    
    # Convert to Python
    N1_py = julia_to_python_network(N1_jl)
    N2_py = julia_to_python_network(N2_jl)
    N_gemini_py = py_src.GeminiNetwork(N1_py, N2_py)
    
    # Create input bounds
    low = fill(-1.0, input_dim)
    high = fill(1.0, input_dim)
    
    # Create verification task (Julia)
    verification_task = create_verification_task(low, high)
    
    # Initialize Julia propagation
    prop_state_jl = PropState(true)
    prepare_prop_state!(prop_state_jl, verification_task)
    
    # Initialize Python propagation
    prop_state_py = py_src.PropState()
    
    # Get initial zonotope from Julia
    initial_zono_jl = prop_state_jl.zono_storage.zonotopes[1].zonotope_proto
    
    # Convert to Python
    initial_zono_py = julia_diffzonotope_to_python(initial_zono_jl)
    
    @info "Comparing initial zonotopes"
    compare_diffzonotopes(initial_zono_jl, initial_zono_py, "Initial", rtol=rtol, atol=atol)
    
    # Propagate layer by layer
    current_zono_jl = initial_zono_jl
    current_zono_py = initial_zono_py
    
    diff_layers_jl = get_layers(N_gemini_jl)
    diff_layers_py = N_gemini_py.diff_layers
    
    for (layer_idx, (diff_layer_jl, diff_layer_py)) in enumerate(zip(diff_layers_jl, diff_layers_py))
        @info "Propagating layer $layer_idx: $(typeof(diff_layer_jl))"
        
        # Julia propagation
        if !VeryDiff.Transformers.has_layer(prop_state_jl, diff_layer_jl)
            VeryDiff.Transformers.init_layer!(prop_state_jl, diff_layer_jl, [prop_state_jl.zono_storage.zonotopes[layer_idx]])
        end
        
        output_ref_jl = VeryDiff.Transformers.get_layer(prop_state_jl, diff_layer_jl)
        input_zonotopes_jl = [current_zono_jl]
        
        # Get bounds cache
        if haskey(prop_state_jl.task_bounds.bounds_cache, diff_layer_jl.layer_idx)
            bounds_cache_jl = prop_state_jl.task_bounds.bounds_cache[diff_layer_jl.layer_idx]
        else
            bounds_cache_jl = VeryDiff.Definitions.BoundsCache()
            prop_state_jl.task_bounds.bounds_cache[diff_layer_jl.layer_idx] = bounds_cache_jl
        end
        
        VeryDiff.Transformers.propagate_layer!(output_ref_jl, diff_layer_jl, input_zonotopes_jl; bounds_cache=bounds_cache_jl)
        current_zono_jl = VeryDiff.Transformers.get_zonotope(output_ref_jl)
        
        # Python propagation
        if !prop_state_py.has_layer(layer_idx - 1)  # Python is 0-indexed
            output_ref_py = N_gemini_py._init_layer(diff_layer_py, current_zono_py, prop_state_py, layer_idx - 1)
            prop_state_py.add_zonotope(output_ref_py)
        else
            output_ref_py = prop_state_py.get_layer_output(layer_idx - 1)
        end
        
        bounds_cache_py = prop_state_py.get_bounds_cache(layer_idx - 1)
        current_zono_py = diff_layer_py.forward(current_zono_py, output_ref_py, bounds_cache_py)
        
        # Compare outputs
        @info "Comparing layer $layer_idx outputs"
        compare_diffzonotopes(current_zono_jl, current_zono_py, "Layer $layer_idx", rtol=rtol, atol=atol)
    end
    
    @info "Layer-by-layer comparison successful!"
end

if "VERYDIFF_COMPARE_FUZZ" in keys(ENV)
    iterations = parse(Int, ENV["VERYDIFF_COMPARE_FUZZ"])
    @testset "Julia-Python Comparison Fuzzer" begin
        i = 1
        @info "Starting comparison fuzzing with $(iterations > 0 ? iterations : "infinite") iterations"
        while true
            ENV["VERYDIFF_TEST_SEED"] = string(rand(1:999999))
            cur_ts = @testset "Comparison iteration $i" begin
                test_seed = parse(Int, ENV["VERYDIFF_TEST_SEED"])
                Random.seed!(test_seed)
                @info "Comparison iteration $i with seed $test_seed"
                
                # Test with different configurations
                @testset "Dense only network" begin
                    test_layer_by_layer_comparison(
                        input_dim=rand(3:8),
                        num_layers=rand(3:7),
                        use_relu=false,
                        rtol=1e-5,
                        atol=1e-6
                    )
                end
                
                @testset "Dense + ReLU network" begin
                    test_layer_by_layer_comparison(
                        input_dim=rand(3:8),
                        num_layers=rand(3:7),
                        use_relu=true,
                        rtol=1e-5,
                        atol=1e-6
                    )
                end
            end
            Test.print_test_results(cur_ts)
            if iterations > 0 && i >= iterations
                break
            end
            i += 1
            # Flush stdout and stderr to ensure logs are written in case of crash
            flush(stdout)
            flush(stderr)
        end
    end
end
