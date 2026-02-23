using VeryDiff
using LinearAlgebra
using Random

"""
    create_random_dense_network(input_dim::Int, layer_dims::Vector{Int})

Create a randomized neural network with only Dense layers using the provided layer dimensions.
Returns both a Network and an OnnxNet representation.
"""
function create_random_dense_network(input_dim::Int, layer_dims::Vector{Int}; relu=false, add_const=false)
    layers = Vector{Node{String}}()
    layer_metadata = Vector{Tuple{String, Int}}()  # Track (layer_name, mutation_type) for synchronization
    cur_dim = input_dim
    layer_count = 0
    prev_output_id = "network_input"
    
    for new_dim in layer_dims
        W = 0.1*randn(Float64, (new_dim, cur_dim))
        b = 0.1*randn(Float64, new_dim)
        # Set some rows to zero
        zero_one_rows = randn(new_dim) .< -2.5
        W[zero_one_rows, :] .= 0.0
        b[zero_one_rows] .= 0.0
        # Set some components to zero
        zero_one_components = randn(size(W)) .< -3.0
        W[zero_one_components] .= 0.0
        layer_count += 1
        input_id = prev_output_id
        output_id = "output_$layer_count"
        push!(layers, ONNXLinear([input_id], [output_id], "dense_$layer_count", W, b))
        prev_output_id = output_id
        
        if add_const
            layer_count += 1
            addconst_input_id = output_id
            addconst_output_id = "output_$layer_count"
            # Create a non-zero constant to add (random values in range [-0.5, 0.5])
            const_vec = 0.1 .* randn(Float64, new_dim)
            # Ensure at least one non-zero element
            if all(c ≈ 0 for c in const_vec)
                const_vec[1] = 0.1
            end
            push!(layers, ONNXAddConst([addconst_input_id], [addconst_output_id], "addconst_$layer_count", const_vec))
            prev_output_id = addconst_output_id
        end
        
        if relu
            layer_count += 1
            relu_input_id = prev_output_id
            relu_output_id = "output_$layer_count"
            push!(layers, ONNXRelu([relu_input_id], [relu_output_id], "relu_$layer_count"))
            prev_output_id = relu_output_id
        end
        cur_dim = new_dim
    end
    
    # Create OnnxNet with appropriate input/output shapes and node structure
    node_dict = Dict{String, Node{String}}()
    for layer in layers
        node_dict[layer.name] = layer
    end
    
    start_nodes = ["dense_1"]
    final_nodes = [layers[end].name]
    input_shapes = Dict("network_input" => (input_dim,))
    output_shapes = Dict(layers[end].outputs[1] => (layer_dims[end],))
    
    onnx_net = OnnxNet(layers, start_nodes, final_nodes, input_shapes, output_shapes)
    
    return onnx_net
end

"""
    create_random_network_mutant(onnx_net :: OnnxNet)

Create a mutated version of the provided ONNX network by mutating every ONNXLinear layer.
Synchronizes mutations between ONNXLinear and corresponding ONNXAddConst layers.
"""
function create_random_network_mutant(onnx_net :: OnnxNet)
    new_layers = Vector{Node{String}}()
    # Store mutation types for each dense layer to sync with addconst
    mutation_map = Dict{String, Int}()
    
    # First pass: determine mutation types for dense layers
    for (node_name, node) in onnx_net.nodes
        if node isa ONNXLinear{String} && startswith(node_name, "dense_")
            mutation_map[node_name] = rand(1:4)
        end
    end
    
    # Second pass: apply mutations
    for (node_name, node) in onnx_net.nodes
        if node isa ONNXLinear{String}
            mutation_type = get(mutation_map, node_name, rand(1:4))
            push!(new_layers, create_random_layer_mutant(node, mutation_type))
        elseif node isa ONNXAddConst{String}
            # Find corresponding dense layer
            dense_name = replace(node_name, "addconst_" => "dense_")
            mutation_type = get(mutation_map, dense_name, rand(1:4))
            push!(new_layers, create_random_layer_mutant(node, mutation_type))
        else
            push!(new_layers, create_random_layer_mutant(node))
        end
    end
    
    # Reconstruct OnnxNet with mutated layers
    start_nodes = onnx_net.start_nodes
    final_nodes = onnx_net.final_nodes
    input_shapes = onnx_net.input_shapes
    output_shapes = onnx_net.output_shapes
    
    return OnnxNet(new_layers, start_nodes, final_nodes, input_shapes, output_shapes)
end

"""
    create_random_layer_mutant(layer :: ONNXLinear, mutation_type :: Int)
Create a mutated version of the provided layer using the specified mutation type.
"""
function create_random_layer_mutant(layer :: ONNXLinear{String}, mutation_type :: Int)
    if mutation_type == 1
        @debug "Independent layer"
        # New random weights and biases
        W_new = 0.1*randn(Float64, size(layer.dense.weight))
        b_new = 0.1*randn(Float64, size(layer.dense.bias))
    elseif mutation_type == 2
        @debug "Zeroed components"
        # Set some weights / biases to zero
        W_new = deepcopy(layer.dense.weight)
        b_new = deepcopy(layer.dense.bias)
        W_mask = randn(size(W_new)) .< -2.0
        b_mask = randn(size(b_new)) .< -2.0
        W_new[W_mask] .= 0.0
        b_new[b_mask] .= 0.0
    elseif mutation_type == 3
        @debug "Pruned rows"
        # Prune some rows
        W_new = deepcopy(layer.dense.weight)
        b_new = deepcopy(layer.dense.bias)
        row_mask = randn(size(W_new, 1)) .< -2.0
        W_new[row_mask, :] .= 0.0
        b_new[row_mask] .= 0.0
    elseif mutation_type == 4
        @debug "Small perturbation"
        # Small random perturbation
        W_new = layer.dense.weight .+ 0.01*randn(Float64, size(layer.dense.weight))
        b_new = layer.dense.bias .+ 0.01*randn(Float64, size(layer.dense.bias))
    else
        error("Unknown mutation type")
    end
    return ONNXLinear(layer.inputs, layer.outputs, layer.name, W_new, b_new, transpose=layer.transpose)
end

"""
    create_random_layer_mutant(layer :: ONNXAddConst, mutation_type :: Int)
Create a mutated version of the AddConst layer using the specified mutation type (synchronized with linear layer).
"""
function create_random_layer_mutant(layer :: ONNXAddConst{String}, mutation_type :: Int)
    if mutation_type == 1
        @debug "Independent AddConst layer"
        # New random constant
        c_new = 0.1 .* randn(Float64, size(layer.c))
        if all(c ≈ 0 for c in c_new)
            c_new[1] = 0.1
        end
    elseif mutation_type == 2
        @debug "Zeroed components in AddConst"
        # Set some components to zero
        c_new = deepcopy(layer.c)
        c_mask = randn(size(c_new)) .< -2.0
        c_new[c_mask] .= 0.0
    elseif mutation_type == 3
        @debug "Zeroed all components in AddConst (pruned)"
        # Zero out the entire constant
        c_new = zeros(size(layer.c))
    elseif mutation_type == 4
        @debug "Small perturbation in AddConst"
        # Small random perturbation
        c_new = layer.c .+ 0.01*randn(Float64, size(layer.c))
    else
        error("Unknown mutation type")
    end
    return ONNXAddConst(layer.inputs, layer.outputs, layer.name, c_new)
end

"""
    create_random_layer_mutant(layer :: Layer)
Create a mutated version of the provided layer by slightly perturbing weights and biases according to different strategies.
"""
function create_random_layer_mutant(layer :: ONNXLinear{String})
    mutation_type = rand(1:4)
    return create_random_layer_mutant(layer, mutation_type)
end

function create_random_layer_mutant(layer :: ONNXRelu{String})
    return deepcopy(layer)
end

function create_random_layer_mutant(layer :: ONNXAddConst{String})
    mutation_type = rand(1:4)
    return create_random_layer_mutant(layer, mutation_type)
end

"""
    make_dense_pair(input_dim::Int, layer_dims::Vector{Int}; identical::Bool=false)

Create two networks sharing the same architecture. When `identical` is true, the weights/biases are identical.
Returns tuples of (Network, OnnxNet) pairs.
"""
function make_dense_pair(input_dim::Int, layer_dims::Vector{Int}; identical::Bool=false, relu=false, add_const=false)
    N1_onnx = create_random_dense_network(input_dim, layer_dims; relu=relu, add_const=add_const)
    if identical
        N2_onnx = deepcopy(N1_onnx)
    else
        N2_onnx = create_random_network_mutant(N1_onnx)
    end
    return N1_onnx, N2_onnx
end

"""
    create_mutant_layers(layers::Vector{Node{String}})

Helper function to create mutated versions of layers.
"""
function create_mutant_layers(layers::Vector{Node{String}})
    new_layers = Vector{Node{String}}()
    for layer in layers
        push!(new_layers, create_random_layer_mutant(layer))
    end
    return new_layers
end

"""
    sample_points_in_hypercube(low::Vector, high::Vector, num_samples::Int)

Sample uniformly from the hypercube defined by `low` and `high` bounds.
"""
function sample_points_in_hypercube(low::Vector, high::Vector, num_samples::Int; secondary_low=nothing, secondary_high=nothing)
    dim = length(low)
    total_dim = dim
    use_secondary = !isnothing(secondary_low) && !isnothing(secondary_high)
    if use_secondary
        @assert length(secondary_low) == length(secondary_high)
        total_dim += length(secondary_low)
    end
    samples = zeros(Float64, total_dim, num_samples)
    for i in 1:num_samples
        samples[1:dim, i] = low .+ (high .- low) .* rand(Float64, dim)
        if use_secondary
            samples[dim+1:end, i] = secondary_low .+ (secondary_high .- secondary_low) .* rand(Float64, length(secondary_low))
        end
    end
    return samples
end

"""
    create_verification_task(low::Vector, high::Vector; with_secondary::Bool=false, secondary_scale::Float64=1.0)

Helper to build a `VerificationTask` for bounds `low`..`high`. When `with_secondary` is true, secondary distances mirror the primary span (scaled by `secondary_scale`).
"""
function create_verification_task(low::Vector, high::Vector; with_secondary::Bool=false, secondary_scale::Float64=1.0)
    mid = (high .+ low) ./ 2
    distance = mid .- low
    non_zero_indices = collect(1:length(low))
    if with_secondary
        dist1 = ones(size(low)) .* secondary_scale
        mid1 = zeros(Float64, length(low))
        dist2 = ones(size(low)) .* secondary_scale
        mid2 = zeros(Float64, length(low))
    else
        dist1 = nothing
        mid1 = nothing
        dist2 = nothing
        mid2 = nothing
    end
    return VerificationTask(
        mid, distance, non_zero_indices,
        dist1, mid1,
        dist2, mid2,
        nothing,
        1.0,
        1.0
    )
end
