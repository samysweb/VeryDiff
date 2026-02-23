function propagate!(N :: GeminiNetwork, P :: PropState)
    max_io_id = maximum(max(maximum(l.inputs), maximum(l.outputs)) for l in get_layers(N))
    if max_io_id > length(P.zono_storage)
        resize_zonotope_storage!(P.zono_storage, max_io_id)
    end
    for diff_layer in get_layers(N)
        # @debug "Propagating layer: $(typeof(diff_layer))"
        input_positions = get_inputs(diff_layer)
        output_positions = get_outputs(diff_layer)
        # @debug "Input Positions: $input_positions"
        inputs = get_zonos_at_pos(input_positions, P)
        if first_pass(P)
            configure_first_usage!(diff_layer.layer_idx, inputs)
        end
        if !zonos_initialized(P, output_positions)
            @assert first_pass(P) "Layer $diff_layer not initialized in PropState! This should only happen during the first pass."
            init_layer!(P, diff_layer, inputs, output_positions)
        end
        input_zonotopes = get_zonotope.(inputs)
        output_positions = get_outputs(diff_layer)
        outputs = get_zonos_at_pos(output_positions, P)
        if haskey(P.task_bounds.bounds_cache, diff_layer.layer_idx)
            # @debug "Bounds Cache: Found for layer index $(diff_layer.layer_idx)"
            bounds_cache = P.task_bounds.bounds_cache[diff_layer.layer_idx]
        else
            # @debug "Bounds Cache: Creating new for layer index $(diff_layer.layer_idx)"
            bounds_cache = BoundsCache()
            P.task_bounds.bounds_cache[diff_layer.layer_idx] = bounds_cache
        end
        # @debug "Processing DiffLayer at index $(diff_layer.layer_idx) with bounds cache initialized=$(bounds_cache.initialized)"
        propagate_layer!(outputs, diff_layer, input_zonotopes; bounds_cache=bounds_cache)
        #@debug "Layer $(diff_layer.layer_idx) output Zonotope ∂Z bounds: $(zono_bounds(output_zonotope_ref.zonotope.∂Z)[1:5,:])"
        #@debug "Layer $(diff_layer.layer_idx) output Zonotope Z₁ bounds: $(zono_bounds(output_zonotope_ref.zonotope.Z₁)[1:5,:])"
    end
    return P
end