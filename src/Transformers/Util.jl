function init_zonotope(output_dim :: Int, input :: Zonotope, influence::Union{Vector{Matrix{Float64}},Nothing}, owned_generators :: Union{Int64, Nothing})
    # Compute new generators
    generators = Vector{Matrix{Float64}}()
    generator_ids = deepcopy(input.generator_ids)
    for g in input.Gs
        new_g = zeros(Float64, output_dim, size(g,2))
        push!(generators, new_g)
    end
    c = zeros(Float64, output_dim)
    return Zonotope(generators, c, influence, generator_ids, owned_generators)
end

function init_layer_dense_z1_z2(output_dim :: Int, input_zono :: DiffZonotope, input_zono_cache :: CachedZonotope, layer_idx :: Int64) :: Tuple{Zonotope,Zonotope}
    # Instantiate Z₁
    # Dense Layer can reuse influence matrix from input (nocopy)
    influence = input_zono.Z₁.influence
    if layer_idx == input_zono_cache.first_usage
        owned_generators = input_zono.Z₁.owned_generators
    else
        owned_generators = nothing
    end
    Z₁ = init_zonotope(output_dim, input_zono.Z₁, influence, owned_generators)
    # Instantiate Z₂
    influence = input_zono.Z₂.influence
    if layer_idx == input_zono_cache.first_usage
        owned_generators = input_zono.Z₂.owned_generators
    else
        owned_generators = nothing
    end
    Z₂ = init_zonotope(output_dim, input_zono.Z₂, influence, owned_generators)
    return Z₁, Z₂
end

function get_selectors(bounds₁, bounds₂, ∂bounds)
    lower₁ = @view bounds₁[:,1]
    upper₁ = @view bounds₁[:,2]
    lower₂ = @view bounds₂[:,1]
    upper₂ = @view bounds₂[:,2]
    ∂lower = @view ∂bounds[:,1]
    ∂upper = @view ∂bounds[:,2]

    dim = length(lower₁)

    zero_diff = @simd_bool_expr dim ((∂upper == 0.0) & (∂lower == 0.0))

    # Compute Phase Behaviour
    check = copy(zero_diff)

    upper₁_leq0 = @simd_bool_expr dim (upper₁ <= 0.0)
    lower₁_geq0 = @simd_bool_expr dim (lower₁ >= 0.0)
    upper₂_leq0 = @simd_bool_expr dim (upper₂ <= 0.0)
    lower₂_geq0 = @simd_bool_expr dim (lower₂ >= 0.0)
    
    neg_neg = @simd_bool_expr dim ((upper₁_leq0) .& (upper₂_leq0) .& .!check)
    check .|= neg_neg
    neg_pos = @simd_bool_expr dim ((upper₁_leq0) .& (lower₂_geq0) .& .!check)
    check .|= neg_pos
    pos_neg = @simd_bool_expr dim ((lower₁_geq0) .& (upper₂_leq0) .& .!check)
    check .|= pos_neg
    pos_pos = @simd_bool_expr dim ((lower₁_geq0) .& (lower₂_geq0) .& .!check)
    check .|= pos_pos
    any_neg = @simd_bool_expr dim ((.!lower₁_geq0) .& (.!upper₁_leq0) .& (upper₂_leq0) .& .!check)
    check .|= any_neg
    neg_any = @simd_bool_expr dim ((upper₁_leq0) .& (.!lower₂_geq0) .& (.!upper₂_leq0) .& .!check)
    check .|= neg_any
    any_pos = @simd_bool_expr dim ((.!lower₁_geq0) .& (.!upper₁_leq0) .& (lower₂_geq0) .& .!check)
    check .|= any_pos
    pos_any = @simd_bool_expr dim ((lower₁_geq0) .& (.!lower₂_geq0) .& (.!upper₂_leq0) .& .!check)
    check .|= pos_any
    any_any = @simd_bool_expr dim ((.!lower₁_geq0) .& (.!upper₁_leq0) .& (.!lower₂_geq0) .& (.!upper₂_leq0) .& .!check)
    check .|= any_any
    @assert all(check) "Not all cases covered: [$(lower₁[.!check]), $(upper₁[.!check])], [$(lower₂[.!check]), $(upper₂[.!check])]"
    return (
        zero_diff,
        neg_neg,
        neg_pos,
        pos_neg,
        pos_pos,
        any_neg,
        neg_any,
        any_pos,
        pos_any,
        any_any
    )
end

function init_relu_zonotope(PS :: PropState, input_zono_cache :: CachedZonotope, input_zono :: Zonotope, new_generators :: Int64, layer_idx :: Int64)
    # Compute new generators
    generators = Matrix{Float64}[]
    generator_ids = SortedVector{Int64}()
    owned_generators = input_zono.owned_generators
    if input_zono_cache.first_usage != layer_idx
        owned_generators = nothing
    end
    for (gid, g) in enumerate(input_zono.Gs)
        if !isnothing(owned_generators) && gid == owned_generators
            cols = size(g,2) + new_generators
            new_g = zeros(Float64, size(g,1), cols)
            # @info "Generator ID $gid: $(size(new_g,2)) columns (owned, adding $new_generators new)"
            push!(generators, new_g)
            push!(generator_ids, input_zono.generator_ids[gid])
        else
            new_g = zeros(Float64, size(g))
            push!(generators, new_g)
            # @info "Generator ID $gid: $(size(new_g,2)) columns"
            push!(generator_ids, input_zono.generator_ids[gid])
        end
    end
    if isnothing(owned_generators)
        owned_generators = get_free_generator_id!(PS)
        new_g = zeros(Float64, size(input_zono.Gs[1],1), new_generators)
        # @info "Generator ID $owned_generators: $(size(new_g,2)) columns (new owned generator)"
        push!(generators, new_g)
        push!(generator_ids, owned_generators)
        owned_generators = length(generator_ids)
    end
    c = zeros(Float64, size(input_zono.Gs[1],1))
    if isnothing(input_zono.influence)
        influence = nothing
    else
        influence = Matrix{Float64}[]
        for (g_id, g) in enumerate(generators)
            if g_id != owned_generators
                push!(influence, input_zono.influence[g_id])
            else
                push!(influence, zeros(Float64, size(input_zono.influence[1],1), size(g,2)))
            end
        end
    end
    return Zonotope(generators, c, influence, generator_ids, owned_generators)
end