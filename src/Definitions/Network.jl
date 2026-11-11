

struct OnnxLayer{LayerIdT} <: Node{LayerIdT}
    layer_index :: Int64
    layer_id :: LayerIdT
    node :: Node{LayerIdT}
    input_ids :: Vector{Int64}
    output_ids :: Vector{Int64}
end

function cleanup_network(layers :: Vector{OnnxLayer{LayerIdT}}) :: Vector{OnnxLayer{LayerIdT}} where {LayerIdT}
    valid_layers = []
    for i in 1:length(layers)
        if layers[i].node isa ONNXLinear{LayerIdT}
            W = layers[i].node.dense.weight
            b = layers[i].node.dense.bias
            f = layers[i].node.dense.σ
            if all(isone.(diag(W))) && all([all(iszero.(diag(W, k))) && all(iszero.(diag(W, -k))) for k in 1:size(W,1)-1]) && all(iszero.(b)) && f == identity
                @info "Removing identity Dense layer at index $i"
                continue
            end
        end
        push!(valid_layers, i)
    end
    print(valid_layers)
    return layers[valid_layers]
end

struct DiffLayer{L1<:Node, Ld<:Node, L2<:Node}
    layer_idx :: Int64
    inputs :: Vector{Int64}
    outputs :: Vector{Int64}
    layer1 :: L1
    diff_layer :: Ld
    layer2 :: L2
    function DiffLayer(layer_idx :: Int64, inputs :: Vector{Int64}, outputs :: Vector{Int64}, layer1 :: L1, diff_layer :: Ld, layer2 :: L2) where {L1<:Node, Ld<:Node, L2<:Node}
        return new{L1,Ld,L2}(layer_idx, inputs, outputs, layer1, diff_layer, layer2)
    end
end

struct ZeroDense{LayerIdT} <: Node{LayerIdT}
end

function topological_sort!(network :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}, node_id :: LayerIdT, visited :: Set{LayerIdT}, sorting :: Dict{LayerIdT, Int64}) where {LayerIdT,NShapeIn,NShapeOut}
    if node_id in visited
        return
    end
    push!(visited, node_id)
    for next_node in network.node_nexts[node_id]
        topological_sort!(network, next_node, visited, sorting)
    end
    sorting[node_id] = length(sorting) + 1
end

function compute_io(
    io_mapping :: Dict{LayerIdT, Int64},
    to_map :: Vector{String}
) :: Vector{Int64} where {LayerIdT}
    result_ids = Int64[]
    for input_id in to_map
        push!(result_ids, io_mapping[input_id])
    end
    return result_ids
end

function sort_network(network :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) :: Tuple{Vector{OnnxLayer{LayerIdT}}, Dict{LayerIdT, Int64}} where {LayerIdT,NShapeIn,NShapeOut}
    @assert length(network.input_shapes) == 1 "Currently only single input networks are supported"
    @assert length(network.output_shapes) == 1 "Currently only single output networks are supported"
    io_map = Dict{LayerIdT, Int64}()
    for (idx, (input_id, _)) in enumerate(network.input_shapes)
        io_map[input_id] = idx
    end
    io_offset = length(io_map)
    sorting = Dict{LayerIdT, Int64}()
    visited = Set{LayerIdT}()
    for input_node in network.start_nodes
        topological_sort!(network, input_node, visited, sorting)
    end
    sorting_inverse = Dict{Int64, LayerIdT}()
    num_nodes = length(sorting)
    # Map i to num_nodes - i + 1
    for (k,v) in sorting
        sorting[k] = num_nodes - v + 1
        sorting_inverse[sorting[k]] = k
    end
    for lid in 1:num_nodes
        node = network.nodes[sorting_inverse[lid]]
        for in_id in node.inputs
            @assert haskey(io_map, in_id)
        end
        for out_id in node.outputs
            io_offset += 1
            io_map[out_id] = io_offset
        end
    end
    for (idx, (output_id, _)) in enumerate(network.output_shapes)
        if !haskey(io_map, output_id)
            io_map[output_id] = idx + io_offset
        end
    end
    layers = OnnxLayer{LayerIdT}[]
    for (node_id, node) in network.nodes
        input_ids = compute_io(io_map, node.inputs)
        output_ids = compute_io(io_map, node.outputs)
        push!(layers, OnnxLayer{LayerIdT}(sorting[node_id], node_id, node, input_ids, output_ids))
    end
    sorted_layers = sort(layers, by = x -> x.layer_index)
    return sorted_layers, io_map
end

function sort_mirror_network(layers :: Vector{OnnxLayer{LayerIdT}}, io_map :: Dict{LayerIdT, Int64}, network :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) :: Vector{Node{LayerIdT}} where {LayerIdT,NShapeIn,NShapeOut}
    sorted_layers = Node{LayerIdT}[]
    for cur_layer in layers
        cur_node = network.nodes[cur_layer.layer_id]
        input_ids = compute_io(io_map, cur_node.inputs)
        output_ids = compute_io(io_map, cur_node.outputs)
        push!(sorted_layers, OnnxLayer{LayerIdT}(cur_layer.layer_index, cur_layer.layer_id, cur_node, input_ids, output_ids))
    end
    return sorted_layers
end


struct GeminiNetwork{LayerIdT}
    inputs :: Dict{LayerIdT, Int64}
    diff_layers :: Vector{DiffLayer}
    function GeminiNetwork(network1 :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}, network2 :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) where {LayerIdT,NShapeIn,NShapeOut}
        n1_layers, io_map = sort_network(network1)
        n2_layers = sort_mirror_network(n1_layers, io_map, network2)
        diff_layers = DiffLayer[]
        if length(n1_layers) > length(n2_layers)
            n1_layers = cleanup_network(network1)
        elseif length(n2_layers) > length(n1_layers)
            n2_layers = cleanup_network(network2)
        end
        @assert length(n1_layers) == length(n2_layers) "Networks have different number of layers after cleanup: $(length(n1_layers)) vs $(length(n2_layers))"
        for (idx, (l1, l2)) in enumerate(zip(n1_layers, n2_layers))
            @assert typeof(l1.node) == typeof(l2.node) "Mismatch in layer types: $(typeof(l1.node)) vs $(typeof(l2.node))"
            @assert l1.input_ids == l2.input_ids "Mismatch in input ids for layers at index $(l1.layer_id) and $(l2.layer_id)"
            if typeof(l1.node) == ONNXLinear{LayerIdT}
                W1 = l1.node.dense.weight
                b1 = l1.node.dense.bias
                W2 = l2.node.dense.weight
                b2 = l2.node.dense.bias
                f1 = l1.node.dense.σ
                f2 = l2.node.dense.σ
                @assert f1 == identity "Unsupported activation function in network1: $f1"
                @assert f2 == identity "Unsupported activation function in network2: $f2"
                @assert size(W1) == size(W2) "Mismatch in weight matrix size: $(size(W1)) vs $(size(W2))"
                @assert size(b1) == size(b2)
                new_W = W1 .- W2
                new_b = b1 .- b2
                if all(iszero.(new_W)) && all(iszero.(new_b))
                    @info "Detected zero difference in Dense layer, replacing with ZeroDense layer."
                    diff_l = ZeroDense{LayerIdT}()
                else
                    diff_l = ONNXLinear(l1.node.inputs, l1.node.outputs, l1.node.name, new_W, new_b)
                end
                push!(diff_layers, DiffLayer(l1.layer_index, l1.input_ids, l1.output_ids, l1.node, diff_l, l2.node))
            elseif typeof(l1.node) == ONNXAddConst{LayerIdT}
                b1 = l1.node.c
                b2 = l2.node.c
                new_b = b1 .- b2
                diff_l = ONNXAddConst(l1.node.inputs, l1.node.outputs, l1.node.name, new_b)
                push!(diff_layers, DiffLayer(l1.layer_index, l1.input_ids, l1.output_ids, l1.node, diff_l, l2.node))
            else
                push!(diff_layers, DiffLayer(l1.layer_index, l1.input_ids, l1.output_ids, l1.node, l1.node, l2.node))
            end
        end
        input_map = Dict{LayerIdT, Int64}()
        for (input_id, idx) in network1.input_shapes
            input_map[input_id] = io_map[input_id]
        end
        return new{LayerIdT}(input_map, diff_layers)
    end
end

function get_inputs(L :: DiffLayer{<:Node,<:Node,<:Node}) :: Vector{Int64}
    return L.inputs
end

function get_outputs(L :: DiffLayer{<:Node,<:Node,<:Node}) :: Vector{Int64}
    return L.outputs
end

function get_layer1(L :: DiffLayer{L1, Ld, L2}) :: L1 where {L1<:Node, Ld<:Node, L2<:Node}
    return L.layer1
end

function get_diff_layer(L :: DiffLayer{L1, Ld, L2}) :: Ld where {L1<:Node, Ld<:Node, L2<:Node}
    return L.diff_layer
end

function get_layer2(L :: DiffLayer{L1, Ld, L2}) :: L2 where {L1<:Node, Ld<:Node, L2<:Node}
    return L.layer2
end

function get_layers(N::GeminiNetwork)
    return N.diff_layers
end

struct Network{LayerIdT,NShapeIn,NShapeOut}
    model :: OnnxNet{LayerIdT,NShapeIn,NShapeOut}
    input_id :: LayerIdT
    output_id :: LayerIdT
end

function (network::Network{LayerIdT,NShapeIn,NShapeOut})(x) where {LayerIdT,NShapeIn,NShapeOut}
    input_data = Dict(network.input_id => x)
    output_data = compute_outputs(network.model, input_data)
    return output_data[network.output_id]
end

function executable_network(model :: OnnxNet{LayerIdT,NShapeIn, NShapeOut}) :: Network where {LayerIdT,NShapeIn,NShapeOut}
    inputs = model.input_shapes
    outputs = model.output_shapes
    @assert length(inputs) == 1 "Currently only single input networks are supported"
    @assert length(outputs) == 1 "Currently only single output networks are supported"
    input_id = first(keys(inputs))
    output_id = first(keys(outputs))
    return Network(model, input_id, output_id)
end
