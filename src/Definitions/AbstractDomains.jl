struct Zonotope
    # Vector of Generator Matrices
    Gs::Vector{Matrix{Float64}}
    # Center of Zonotope
    c::Vector{Float64}
    # Influence Matrix (necessary for input splitting heuristic)
    # TODO: Move this to PropState at some point?
    influence::Union{Vector{Matrix{Float64}},Nothing}
    # ID of generator matrices
    # must be same length as G
    generator_ids :: SortedVector{Int64}
    # Index of generator matrix owned by Zonotope
    # We may only append columns to the owned generator matrix
    # This Zonotope must be the only active Zonotope that modifies this generator matrix
    # If no owned generator matrix has been allocated, this field is nothing
    owned_generators :: Union{Int64, Nothing}
    function Zonotope(
        Gs::Vector{Matrix{Float64}},
        c::Vector{Float64},
        influence::Union{Vector{Matrix{Float64}},Nothing},
        generator_ids :: SortedVector{Int64},
        owned_generators :: Union{Int64, Nothing}
    )
        @assert length(Gs) == length(generator_ids) "Number of generator matrices must match number of generator IDs"
        @assert isnothing(owned_generators) || owned_generators <= length(generator_ids) "Owned generator index out of bounds"
        return new(Gs, c, influence, generator_ids, owned_generators)
    end
end

# Z₂ = Z₁ - ∂Z

struct DiffZonotope
    Z₁::Zonotope
    Z₂::Zonotope
    ∂Z::Zonotope
end

mutable struct BoundsCache
    initialized :: Bool
    lower₁ :: Union{Vector{Float64}, Nothing}
    upper₁ :: Union{Vector{Float64}, Nothing}
    lower₂ :: Union{Vector{Float64}, Nothing}
    upper₂ :: Union{Vector{Float64}, Nothing}
    ∂lower :: Union{Vector{Float64}, Nothing}
    ∂upper :: Union{Vector{Float64}, Nothing}
    function BoundsCache()
        return new(false, nothing,nothing,nothing,nothing,nothing,nothing)
    end
end

mutable struct CachedZonotope
    zonotope_proto :: DiffZonotope
    zonotope :: Union{Nothing, DiffZonotope}
    first_usage :: Union{Nothing,Int64} # The first usage of this Zonotope has permission to own generators
    function CachedZonotope(zono_proto :: DiffZonotope, first_usage :: Union{Nothing,Int64})
        return new(zono_proto, nothing, first_usage)
    end
end

struct ZonotopeStorage
    zonotopes :: Vector{Union{Nothing,CachedZonotope}}
end

function length(zs :: ZonotopeStorage) :: Int64
    return length(zs.zonotopes)
end

function resize_zonotope_storage!(zs :: ZonotopeStorage, new_size :: Int64)
    current_size = length(zs.zonotopes)
    @assert new_size > current_size
    for _ in 1:(new_size - current_size)
        push!(zs.zonotopes, nothing)
    end
end

# WARNING: This method is dangerous! We are generating Matrix pointers to sub-matrices of existing matrices.
# Make sure that the original matrices are not deallocated while these views are in use!
# Also make sure that we do not assume more columns than actually allocated in the original matrices!
function get_zonotope!(
    zono :: CachedZonotope,
    needed_columns₁ :: Vector{Int64},
    needed_columns₂ :: Vector{Int64},
    needed_columns_∂ :: Vector{Int64}) :: DiffZonotope
    # Check if we already have the needed columns
    #@assert all(needed_columns₁ .<= size.(zono.zonotope_proto.Z₁.Gs,2)) "Requested more generator columns than available in Zonotope 1!"
    #@assert all(needed_columns₂ .<= size.(zono.zonotope_proto.Z₂.Gs,2)) "Requested more generator columns than available in Zonotope 2!"
    #@assert all(needed_columns_∂ .<= size.(zono.zonotope_proto.∂Z.Gs,2)) "Requested more generator columns than available in Differential Zonotope!"
    # Create view based new Zonotope
    for (i, needed_columns) in enumerate(needed_columns₁)
        A = zono.zonotope_proto.Z₁.Gs[i]
        @assert needed_columns <= size(A,2) "Requested $needed_columns columns, but only $(size(A,2)) available in generator matrix $i of Z₁!"
        #zono.zonotope.Z₁.Gs[i] = @view A[:, 1:needed_columns]
        zono.zonotope.Z₁.Gs[i] = unsafe_wrap(Matrix{Float64}, pointer(A), (size(A,1), needed_columns); own=false)
        if !isnothing(zono.zonotope_proto.Z₁.influence)
            B = zono.zonotope_proto.Z₁.influence[i]
            #zono.zonotope.Z₁.influence[i] = @view B[:, 1:needed_columns]
            zono.zonotope.Z₁.influence[i] = unsafe_wrap(Matrix{Float64}, pointer(B), (size(B,1), needed_columns); own=false)
        else
            @assert isnothing(zono.zonotope_proto.Z₁.influence) "Zonotope influence should be nothing if not present in prototype!"
        end
    end
    for (i, needed_columns) in enumerate(needed_columns₂)
        A = zono.zonotope_proto.Z₂.Gs[i]
        @assert needed_columns <= size(A,2) "Requested $needed_columns columns, but only $(size(A,2)) available in generator matrix $i of Z₂!"
        #zono.zonotope.Z₂.Gs[i] = @view A[:, 1:needed_columns]
        zono.zonotope.Z₂.Gs[i] = unsafe_wrap(Matrix{Float64}, pointer(A), (size(A,1), needed_columns); own=false)
        if !isnothing(zono.zonotope_proto.Z₂.influence)
            B = zono.zonotope_proto.Z₂.influence[i]
            #zono.zonotope.Z₂.influence[i] = @view B[:, 1:needed_columns]
            zono.zonotope.Z₂.influence[i] = unsafe_wrap(Matrix{Float64}, pointer(B), (size(B,1), needed_columns); own=false)
        else
            @assert isnothing(zono.zonotope.Z₂.influence) "Zonotope influence should be nothing if not present in prototype!"
        end
    end
    for (i, needed_columns) in enumerate(needed_columns_∂)
        A = zono.zonotope_proto.∂Z.Gs[i]
        @assert needed_columns <= size(A,2) "Requested $needed_columns columns, but only $(size(A,2)) available in generator matrix $i of ∂Z!"
        #zono.zonotope.∂Z.Gs[i] = @view A[:, 1:needed_columns]
        zono.zonotope.∂Z.Gs[i] = unsafe_wrap(Matrix{Float64}, pointer(A), (size(A,1), needed_columns); own=false)
        if !isnothing(zono.zonotope_proto.∂Z.influence)
            B = zono.zonotope_proto.∂Z.influence[i]
            #zono.zonotope.∂Z.influence[i] = @view B[:, 1:needed_columns]
            zono.zonotope.∂Z.influence[i] = unsafe_wrap(Matrix{Float64}, pointer(B), (size(B,1), needed_columns); own=false)
        else
            @assert isnothing(zono.zonotope.∂Z.influence) "Zonotope influence should be nothing if not present in prototype!"
        end
    end
    # Set all generators and centers to zero initially
    for g in zono.zonotope.Z₁.Gs
        fill!(g, 0.0)
    end
    for g in zono.zonotope.Z₂.Gs
        fill!(g, 0.0)
    end
    for g in zono.zonotope.∂Z.Gs
        fill!(g, 0.0)
    end
    fill!(zono.zonotope.Z₁.c, 0.0)
    fill!(zono.zonotope.Z₂.c, 0.0)
    fill!(zono.zonotope.∂Z.c, 0.0)
    return zono.zonotope
end