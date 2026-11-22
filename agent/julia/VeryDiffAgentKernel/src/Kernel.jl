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
    infeasible::Union{Nothing,Dict{String,Any}}   # witness that the node's concrete set is empty (propagation stopped)
end

"""Group the statements of in-scope lemmas and of the node's case-split constraints by ReLU layer."""
function seeds_for(S::Session, lemma_ids::Vector{Int}, constraints::Vector{LemmaStatement}=LemmaStatement[])
    seeds = Dict{Int,Vector{LemmaStatement}}()
    for L in lemma_ids
        st = getlemma(S, L).stmt
        push!(get!(seeds, st.relu_layer, LemmaStatement[]), st)
    end
    for st in constraints
        push!(get!(seeds, st.relu_layer, LemmaStatement[]), st)
    end
    return seeds
end

"""Intersect the cache with the fresh zonotope bounds of the ReLU input (exactly what VeryDiff's ReLU transformer
does itself; doing it here first lets the kernel tighten and test for emptiness before the transformer runs)."""
function intersect_fresh!(bc::BoundsCache, Zi)
    b1 = zono_bounds(Zi.Z₁); b2 = zono_bounds(Zi.Z₂); bd = zono_bounds(Zi.∂Z)
    if !bc.initialized
        bc.lower₁ = b1[:, 1]; bc.upper₁ = b1[:, 2]; bc.lower₂ = b2[:, 1]; bc.upper₂ = b2[:, 2]
        bc.∂lower = bd[:, 1]; bc.∂upper = bd[:, 2]; bc.initialized = true
    else
        bc.lower₁ .= max.(bc.lower₁, b1[:, 1]); bc.upper₁ .= min.(bc.upper₁, b1[:, 2])
        bc.lower₂ .= max.(bc.lower₂, b2[:, 1]); bc.upper₂ .= min.(bc.upper₂, b2[:, 2])
        bc.∂lower .= max.(bc.∂lower, bd[:, 1]); bc.∂upper .= min.(bc.∂upper, bd[:, 2])
    end
end

"""Consistency tightening with z1 = z2 + d (one round, rounded outward)."""
function csb_tighten!(bc::BoundsCache)
    l1, u1, l2, u2, dl, du = bc.lower₁, bc.upper₁, bc.lower₂, bc.upper₂, bc.∂lower, bc.∂upper
    l1 .= max.(l1, prevfloat.(l2 .+ dl)); u1 .= min.(u1, nextfloat.(u2 .+ du))
    l2 .= max.(l2, prevfloat.(l1 .- du)); u2 .= min.(u2, nextfloat.(u1 .- dl))
    dl .= max.(dl, prevfloat.(l1 .- u2)); du .= min.(du, nextfloat.(u1 .- l2))
end

# ------------------------------------------------------------------ zonotope contraction (case-split constraints)
# A case-split constraint lo ≤ v_j ≤ hi on a ReLU input holds for every concrete execution of the node. With
# v_j = c + g·ε over the shared noise symbols, it restricts ε to a smaller box (interval constraint propagation);
# substituting ε_i = m_i + r_i ε'_i (ε' ∈ [-1,1]) in every live zonotope keeps them sound for the node's concrete
# set and tightens everything downstream (the "active" branch is no longer the identity over the whole zonotope).

"""Tighten the box [lo,hi] with g·ε ≤ rhs. Returns false if infeasible (with a rounding allowance)."""
function contract_row!(lo::Vector{Float64}, hi::Vector{Float64}, g::Vector{Float64}, rhs::Float64)
    isfinite(rhs) || return true
    slack = 1e-12 * (abs(rhs) + sum(abs, g) + 1.0)
    r = rhs + slack
    mins = [gi > 0 ? gi * lo[i] : gi * hi[i] for (i, gi) in enumerate(g)]
    tot = sum(mins)
    tot > r && return false
    for (i, gi) in enumerate(g)
        abs(gi) < 1e-300 && continue
        b = (r - (tot - mins[i])) / gi
        if gi > 0
            b = nextfloat(nextfloat(b)) + 1e-12 * abs(b)
            b < hi[i] && (hi[i] = max(b, lo[i] - 0.0))
        else
            b = prevfloat(prevfloat(b)) - 1e-12 * abs(b)
            b > lo[i] && (lo[i] = min(b, hi[i]))
        end
    end
    return true
end

"""Contract the noise box of the live zonotopes with the constraints `stmts` on the input `Zi` of a ReLU layer
and rescale every live zonotope in `P`. Returns nothing, or a witness Dict if the constraints are infeasible."""
function contract_zonos!(P::PropState, Zi, stmts::Vector{LemmaStatement})
    off, m = var_layout([Zi.∂Z, Zi.Z₁, Zi.Z₂])
    lo = fill(-1.0, m); hi = fill(1.0, m)
    rows = Tuple{AffRow,Float64,Float64}[]
    for st in stmts, (k, j) in enumerate(st.neurons)
        if !isnothing(st.n1lo); push!(rows, (affrow(Zi.Z₁, j, off, m), st.n1lo[k], st.n1hi[k])); end
        if !isnothing(st.n2lo); push!(rows, (affrow(Zi.Z₂, j, off, m), st.n2lo[k], st.n2hi[k])); end
        if isfinite(st.dlo[k]) || isfinite(st.dhi[k]); push!(rows, (affrow(Zi.∂Z, j, off, m), st.dlo[k], st.dhi[k])); end
    end
    for _ in 1:2, (row, l, u) in rows
        contract_row!(lo, hi, row.g, u - row.c) || return Dict{String,Any}("contraction" => true)
        contract_row!(lo, hi, .-row.g, row.c - l) || return Dict{String,Any}("contraction" => true)
    end
    any(lo .> hi) && return Dict{String,Any}("contraction" => true)
    changed = Dict{Int,Vector{Tuple{Int,Float64,Float64}}}()   # block id => (col, m, r)
    ids = sort(collect(keys(off)))
    for (t, id) in enumerate(ids)
        o = off[id]; w = (t < length(ids) ? off[ids[t+1]] : m) - o
        for c in 1:w
            i = o + c
            (lo[i] == -1.0 && hi[i] == 1.0) && continue
            mm = lo[i] / 2 + hi[i] / 2
            r = nextfloat(max(hi[i] - mm, mm - lo[i]))       # [mm - r, mm + r] ⊇ [lo, hi]
            push!(get!(changed, id, Tuple{Int,Float64,Float64}[]), (c, mm, r))
        end
    end
    isempty(changed) && return nothing
    rescale_storage!(P, changed)
    return nothing
end

function rescale_storage!(P::PropState, changed)
    zs = Any[]; seenz = Set{UInt}()
    for cz in P.zono_storage.zonotopes
        (isnothing(cz) || isnothing(cz.zonotope)) && continue
        for Z in (cz.zonotope.Z₁, cz.zonotope.Z₂, cz.zonotope.∂Z)
            objectid(Z) in seenz && continue
            push!(seenz, objectid(Z)); push!(zs, Z)
        end
    end
    for Z in zs, (bi, id) in enumerate(collect(Z.generator_ids))       # centres first, with the original G
        haskey(changed, id) || continue
        G = Z.Gs[bi]
        for (c, mm, _) in changed[id]
            c <= size(G, 2) && mm != 0.0 && (Z.c .+= mm .* @view G[:, c])
        end
    end
    seeng = Set{UInt}()
    for Z in zs, (bi, id) in enumerate(collect(Z.generator_ids))
        haskey(changed, id) || continue
        G = Z.Gs[bi]
        objectid(G) in seeng && continue
        push!(seeng, objectid(G))
        for (c, _, r) in changed[id]
            c <= size(G, 2) && (G[:, c] .*= r)
        end
    end
end

const INFEASIBLE_RTOL = 1e-9
"""First (var, neuron, gap) with lower − upper > tol·(1 + max(|lower|, |upper|)), else nothing."""
function infeasible_entry(bc::BoundsCache)
    for (v, lo, hi) in (("z1", bc.lower₁, bc.upper₁), ("z2", bc.lower₂, bc.upper₂), ("d", bc.∂lower, bc.∂upper))
        for j in eachindex(lo)
            l, u = lo[j], hi[j]
            (isfinite(l) && isfinite(u)) || continue
            if l - u > INFEASIBLE_RTOL * (1 + max(abs(l), abs(u)))
                return (v, j, l - u)
            end
        end
    end
    return nothing
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
function propagate_seeded!(N, P::PropState, seeds::Dict{Int,Vector{LemmaStatement}}, stop::Union{Nothing,Int};
                           csb::Bool=false, infeasible::Ref{Any}=Ref{Any}(nothing),
                           contract::Dict{Int,Vector{LemmaStatement}}=Dict{Int,Vector{LemmaStatement}}())
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
        if haskey(contract, diff_layer.layer_idx)
            w = contract_zonos!(P, input_zonotopes[1], contract[diff_layer.layer_idx])
            if !isnothing(w)
                w["layer"] = diff_layer.layer_idx
                infeasible[] = w
                break
            end
        end
        outputs = get_zonos_at_pos(output_positions, P)
        bounds_cache = get!(P.task_bounds.bounds_cache, diff_layer.layer_idx) do
            BoundsCache()
        end
        if haskey(seeds, diff_layer.layer_idx)
            diff_layer.layer1 isa ONNXRelu || kerr("lemma", "lemma layer $(diff_layer.layer_idx) is not a ReLU layer")
            apply_seeds!(bounds_cache, length(input_zonotopes[1].Z₁.c), seeds[diff_layer.layer_idx])
        end
        if diff_layer.layer1 isa ONNXRelu
            intersect_fresh!(bounds_cache, input_zonotopes[1])
            # Seeded layers (case splits, lemmas) always get the cross-network tightening, so that a bound on
            # one network's pre-activation also tightens the other one's via d = z1 − z2.
            (csb || haskey(seeds, diff_layer.layer_idx)) && csb_tighten!(bounds_cache)
            e = infeasible_entry(bounds_cache)
            if !isnothing(e)
                infeasible[] = Dict{String,Any}("layer" => diff_layer.layer_idx, "var" => e[1], "neuron" => e[2], "gap" => e[3])
                break
            end
        end
        propagate_layer!(outputs, diff_layer, input_zonotopes; bounds_cache=bounds_cache)
        push!(done, diff_layer.layer_idx)
    end
    return done
end

"""Run one propagation of `box` from (a deepcopy of) `cache_in` with the given lemma seeds."""
function run_propagation(S::Session, box::Box, cache_in::TaskBounds, lemma_ids::Vector{Int},
                         stop::Union{Nothing,Int}; verification_status=nothing,
                         constraints::Vector{LemmaStatement}=LemmaStatement[])
    VeryDiff.ONE_SIDED_DIFF_FALLBACK[] = S.fallback
    cache = deepcopy(cache_in)
    task = to_task(box, S.active_dims, cache; verification_status=verification_status)
    ps = PropState(true)
    prepare_prop_state!(ps, task)
    ps.task_bounds === cache || kerr("internal", "prop state does not use the owned cache")
    Zin = ps.zono_storage.zonotopes[1].zonotope
    infeas = Ref{Any}(nothing)
    done = propagate_seeded!(S.N, ps, seeds_for(S, lemma_ids, constraints), stop; csb=S.csb, infeasible=infeas,
                             contract=seeds_for(S, Int[], constraints))
    S.n_propagations += 1
    if !isnothing(infeas[])
        inconsistent = [k for (k, bc) in cache.bounds_cache if cache_inconsistent(bc)]
        return PropResult(ps, task, Zin, nothing, cache, sort(inconsistent), false, done, stop, infeas[])
    end
    if isnothing(stop)
        Zout = ps.zono_storage.zonotopes[get_outputs(get_layers(S.N)[end])[1]].zonotope
    else
        L = only(l for l in get_layers(S.N) if l.layer_idx == stop)
        Zout = ps.zono_storage.zonotopes[get_inputs(L)[1]].zonotope
    end
    inconsistent = [k for (k, bc) in cache.bounds_cache if cache_inconsistent(bc)]
    nonfinite = any(cache_nonfinite, values(cache.bounds_cache)) ||
        any(isnan, zono_bounds(Zout.∂Z)) || any(isnan, zono_bounds(Zout.Z₁)) || any(isnan, zono_bounds(Zout.Z₂))
    return PropResult(ps, task, Zin, Zout, cache, sort(inconsistent), nonfinite, done, stop, nothing)
end

layer_by_idx(S::Session, k::Int) = (i = findfirst(l -> l.layer_idx == k, get_layers(S.N));
    isnothing(i) ? kerr("bad_layer", "no layer with index $k") : get_layers(S.N)[i])
