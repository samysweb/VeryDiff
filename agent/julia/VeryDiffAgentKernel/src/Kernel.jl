# Propagation glue: a seeded copy of VeryDiff's `propagate!` loop.
#
# Every propagation uses a *fresh* PropState initialised from the task itself. This keeps
# generator allocation consistent (init uses the task's own fresh bounds; the cached bounds
# are always an intersection with them) and makes nodes independent of each other.

struct PropResult
    ps::PropState
    task::VerificationTask
    Zin::Any
    Zout::Any                         # output zonotope, or the input of the stop layer
    cache::TaskBounds                 # post-propagation cache (owned by the caller)
    inconsistent::Vector{Int}         # ReLU layers whose cache has lower > upper somewhere
    nonfinite::Bool                   # NaN/Inf in Zout bounds or in any used cache
    layers_done::Vector{Int}
    stop::Union{Nothing,Int}
end

"""Group the statements of in-scope lemmas by ReLU layer."""
function seeds_for(S::Session, lemma_ids::Vector{Int})
    seeds = Dict{Int,Vector{LemmaStatement}}()
    for L in lemma_ids
        st = getlemma(S, L).stmt
        push!(get!(seeds, st.relu_layer, LemmaStatement[]), st)
    end
    return seeds
end

"""Seed `bc` (cache of a ReLU layer with input width `n`) with lemma bounds.
An uninitialised cache is first set to [-Inf, +Inf] and marked initialised, otherwise the
ReLU transformer would overwrite the seed with the fresh bounds."""
function apply_seeds!(bc::BoundsCache, n::Int, stmts::Vector{LemmaStatement})
    if !bc.initialized
        bc.lower₁ = fill(-Inf, n); bc.upper₁ = fill(Inf, n)
        bc.lower₂ = fill(-Inf, n); bc.upper₂ = fill(Inf, n)
        bc.∂lower = fill(-Inf, n); bc.∂upper = fill(Inf, n)
        bc.initialized = true
    end
    for s in stmts
        all(1 .<= s.neurons .<= n) || kerr("lemma", "lemma neuron index out of range for layer $(s.relu_layer)")
        bc.∂lower[s.neurons] .= max.(bc.∂lower[s.neurons], s.dlo)
        bc.∂upper[s.neurons] .= min.(bc.∂upper[s.neurons], s.dhi)
        if !isnothing(s.n1lo)
            bc.lower₁[s.neurons] .= max.(bc.lower₁[s.neurons], s.n1lo)
            bc.upper₁[s.neurons] .= min.(bc.upper₁[s.neurons], s.n1hi)
        end
        if !isnothing(s.n2lo)
            bc.lower₂[s.neurons] .= max.(bc.lower₂[s.neurons], s.n2lo)
            bc.upper₂[s.neurons] .= min.(bc.upper₂[s.neurons], s.n2hi)
        end
    end
end

cache_inconsistent(bc::BoundsCache) = bc.initialized &&
    (any(bc.lower₁ .> bc.upper₁) || any(bc.lower₂ .> bc.upper₂) || any(bc.∂lower .> bc.∂upper))
cache_nonfinite(bc::BoundsCache) = bc.initialized &&
    any(x -> isnan(x), vcat(bc.lower₁, bc.upper₁, bc.lower₂, bc.upper₂, bc.∂lower, bc.∂upper))

"""
propagate_seeded!(N, P, seeds, stop) — copy of `VeryDiff.Transformers.propagate!` that
(1) seeds ReLU bounds caches from in-scope lemmas before propagating that layer and
(2) optionally stops *before* layer `stop` (prefix propagation for lemma obligations).
"""
function propagate_seeded!(N, P::PropState, seeds::Dict{Int,Vector{LemmaStatement}}, stop::Union{Nothing,Int})
    max_io_id = maximum(max(maximum(l.inputs), maximum(l.outputs)) for l in get_layers(N))
    if max_io_id > length(P.zono_storage.zonotopes)
        resize_zonotope_storage!(P.zono_storage, max_io_id)
    end
    done = Int[]
    for diff_layer in get_layers(N)
        if !isnothing(stop) && diff_layer.layer_idx == stop
            break
        end
        input_positions = get_inputs(diff_layer)
        output_positions = get_outputs(diff_layer)
        inputs = get_zonos_at_pos(input_positions, P)
        if first_pass(P)
            configure_first_usage!(diff_layer.layer_idx, inputs)
        end
        if !zonos_initialized(P, output_positions)
            first_pass(P) || kerr("internal", "layer not initialised outside first pass")
            init_layer!(P, diff_layer, inputs, output_positions)
        end
        input_zonotopes = get_zonotope.(inputs)
        outputs = get_zonos_at_pos(output_positions, P)
        bounds_cache = get!(P.task_bounds.bounds_cache, diff_layer.layer_idx) do
            BoundsCache()
        end
        if haskey(seeds, diff_layer.layer_idx)
            diff_layer.layer1 isa ONNXRelu || kerr("lemma", "lemma layer $(diff_layer.layer_idx) is not a ReLU layer")
            apply_seeds!(bounds_cache, length(input_zonotopes[1].Z₁.c), seeds[diff_layer.layer_idx])
        end
        propagate_layer!(outputs, diff_layer, input_zonotopes; bounds_cache=bounds_cache)
        push!(done, diff_layer.layer_idx)
    end
    return done
end

"""Run one propagation of `box` from (a deepcopy of) `cache_in` with the given lemma seeds."""
function run_propagation(S::Session, box::Box, cache_in::TaskBounds, lemma_ids::Vector{Int},
                         stop::Union{Nothing,Int}; verification_status=nothing)
    cache = deepcopy(cache_in)
    task = to_task(box, S.active_dims, cache; verification_status=verification_status)
    ps = PropState(true)
    prepare_prop_state!(ps, task)
    ps.task_bounds === cache || kerr("internal", "prop state does not use the owned cache")
    Zin = ps.zono_storage.zonotopes[1].zonotope
    done = propagate_seeded!(S.N, ps, seeds_for(S, lemma_ids), stop)
    S.n_propagations += 1
    if isnothing(stop)
        Zout = ps.zono_storage.zonotopes[get_outputs(get_layers(S.N)[end])[1]].zonotope
    else
        L = only(l for l in get_layers(S.N) if l.layer_idx == stop)
        Zout = ps.zono_storage.zonotopes[get_inputs(L)[1]].zonotope
    end
    inconsistent = [k for (k, bc) in cache.bounds_cache if cache_inconsistent(bc)]
    nonfinite = any(cache_nonfinite, values(cache.bounds_cache)) ||
        any(isnan, zono_bounds(Zout.∂Z)) || any(isnan, zono_bounds(Zout.Z₁)) || any(isnan, zono_bounds(Zout.Z₂))
    return PropResult(ps, task, Zin, Zout, cache, sort(inconsistent), nonfinite, done, stop)
end

layer_by_idx(S::Session, k::Int) = (i = findfirst(l -> l.layer_idx == k, get_layers(S.N));
    isnothing(i) ? kerr("bad_layer", "no layer with index $k") : get_layers(S.N)[i])
