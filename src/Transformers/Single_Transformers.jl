function propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXLinear{S1}, inputs :: Vector{Zonotope}) where {S1}
    @assert length(inputs) == 1 "Dense layer should have exactly one input"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin)
end

function propagate_layer!(ZoutRef :: Zonotope, L :: ONNXLinear{S1}, Zin :: Zonotope) where {S1}
    # Zout must have exactly the same ids as Zin
    # @assert all(ZoutRef.generator_ids .== Zin.generator_ids) "Zonotope generator IDs do not match during Dense propagation!"
    for i in 1:length(Zin.Gs)
        mul!(ZoutRef.Gs[i], L.dense.weight, Zin.Gs[i])
    end
    mul!(ZoutRef.c, L.dense.weight, Zin.c)
    ZoutRef.c .+= L.dense.bias
end

function propagate_layer!(ZoutRef :: Zonotope, L :: ONNXAddConst{S1}, Zin :: Zonotope) where {S1}
    # Zout must have exactly the same ids as Zin
    # @assert all(ZoutRef.generator_ids .== Zin.generator_ids) "Zonotope generator IDs do not match during Dense propagation!"
    for i in 1:length(Zin.Gs)
        ZoutRef.Gs[i] .= Zin.Gs[i]
    end
    ZoutRef.c .= Zin.c .+ L.c
end

function get_slope(l,u, alpha)
    if u <= 0
        return 0.0
    elseif l >= 0
        return 1.0
    else
        return alpha
    end
end

function propagate_layer!(ZoutRefVec :: Vector{Zonotope}, L :: ONNXRelu{S}, inputs :: Vector{Zonotope}; lower=nothing, upper=nothing) where {S}
    @assert length(inputs) == 1 "Dense layer should have exactly one input"
    @assert length(ZoutRefVec) == 1 "Dense layer should have exactly one output"
    ZoutRef = ZoutRefVec[1]
    Zin = inputs[1]
    return propagate_layer!(ZoutRef, L, Zin; lower=lower, upper=upper)
end

function propagate_layer!(ZoutRef :: Zonotope, _L :: ONNXRelu{S}, Zin :: Zonotope; lower=nothing, upper=nothing) where {S}
    if isnothing(lower) || isnothing(upper)
        bounds = zono_bounds(Zin)
        lower = @view bounds[:,1]
        upper = @view bounds[:,2]
    end

    dim = length(lower)
    crossing = @simd_bool_expr dim ((lower < 0.0) & (upper > 0.0))
    α = clamp.(upper./(upper.-lower),0.0,1.0)
    # Use is_onesided to compute 
    λ = ifelse.(crossing, α, ifelse.(lower .>= 0.0, 1.0, 0.0))

    new_gens = count(crossing)
    
    γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

    ZoutRef.c .= λ .* Zin.c .+ crossing.*γ

    indices = intersect_indices(ZoutRef.generator_ids, Zin.generator_ids)
    if VeryDiff.NEW_HEURISTIC[]
        influence_new = ZoutRef.influence
        column_pos = size(influence_new[ZoutRef.owned_generators],2) - new_gens + 1
        # @debug "Adding $new_gens new columns at position $column_pos to influence matrix of owned generator ID $(ZoutRef.generator_ids[ZoutRef.owned_generators])"
        # @debug "Sizes of influence matrices: $([size(inf) for inf in Zin.influence])"
        # Other influence matrices remain the same
        # Only need to update the owned generator influence matrix
        if !isnothing(Zin.owned_generators) && Zin.owned_generators == attempt_find_index_position(Zin.generator_ids, ZoutRef.generator_ids[ZoutRef.owned_generators])
            influence_new[ZoutRef.owned_generators][:, 1:column_pos-1] .= Zin.influence[Zin.owned_generators]
        end
        # @debug "Size of owned influence matrix after copy: $(size(influence_new[ZoutRef.owned_generators]))"
        influence_new[ZoutRef.owned_generators][:,column_pos:end] .= 0.0
        bounds_range = upper[crossing] .- lower[crossing]
        @inbounds for (idx, g) in enumerate(Zin.Gs)
            influence_new[ZoutRef.owned_generators][:,column_pos:end] .+= Zin.influence[idx] * abs.((@view g[crossing,:]) ./ bounds_range)'
        end
    else
        influence_new = Zin.influence
    end

    num_new_gens = count(crossing)

    updateGeneratorsMul!(ZoutRef.Gs, indices, Zin.Gs, λ, :)
    ZoutRef.Gs[ZoutRef.owned_generators][:,(end-num_new_gens+1):end] .= 0.0
    generator_offset = size(ZoutRef.Gs[ZoutRef.owned_generators],2) - num_new_gens
    A = ZoutRef.Gs[ZoutRef.owned_generators]
    @inbounds for (i, row) in enumerate(findall(crossing))
        A[row, (generator_offset + i)] = abs(γ[row])
    end
end