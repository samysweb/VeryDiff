#!/usr/bin/env julia
#
# check_certificate.jl — independent replay checker for VeryDiff-Agent certificates.
#
# Usage:
#   julia --project=agent/julia/VeryDiffAgentKernel agent/checker/check_certificate.jl CERT NET1 NET2 SPEC [--verbose]
#
# Output: "ACCEPT <status> ..." (exit code 0) or "REJECT <reason>" (exit code 1).
#
# What this checker trusts:
#   - VeryDiff's layer transformers (init_layer!, propagate_layer!), zono_bounds and the epsilon property.
#   - VNNLib.jl for ONNX and VNNLIB parsing, and compute_all_outputs for concrete evaluation.
#   - Float64 arithmetic.
#
# What this checker does NOT trust:
#   - Any value in the certificate that it can compute itself. It recomputes the spec hash, every region,
#     every propagation, every bound and the closure of the proof tree.
#   - The kernel. The checker does not load the kernel module. It contains its own implementation of boxes,
#     cache seeding, contraction, dual-tightened bounds, LP rows and closure.
#
# The certificate format is described in agent/docs/certificate_format.md.
#
# File structure:
#   1. Errors                      8. Propagation of one node
#   2. Numbers and provenance      9. LP evidence
#   3. Boxes                      10. Reading the certificate
#   4. Constraints                11. Replay of the proof tree
#   5. Zonotope rows              12. Closure and result
#   6. Bounds caches              13. Command line
#   7. Contraction and dual bounds

module CertChecker

using LinearAlgebra
using SHA
using JSON
using CodecZlib
using VNNLib
using VNNLib.OnnxParser: ONNXRelu, ONNXSoftmax, compute_all_outputs
using VeryDiff
using VeryDiff.Definitions: VerificationTask, PropState, BoundsCache, TaskBounds, get_layers, get_inputs,
    get_outputs, get_zonos_at_pos, first_pass, configure_first_usage!, zonos_initialized,
    prepare_prop_state!, resize_zonotope_storage!, get_zonotope, zono_bounds, sort_network, executable_network
using VeryDiff.Transformers: init_layer!, propagate_layer!


# =====================================================================================================
# 1. Errors
# =====================================================================================================

"""A certificate is not valid. The message tells the reason."""
struct Reject <: Exception
    msg::String
end

reject(msg) = throw(Reject(msg))

"""Reject the certificate with `msg` if `condition` is false."""
need(condition, msg) = condition || reject(msg)


# =====================================================================================================
# 2. Numbers and provenance
# =====================================================================================================

"""Read a number. JSON cannot encode infinity, so the certificate writes it as "Infinity" or "-Infinity"."""
function decode_number(x)
    x isa Number && return Float64(x)
    s = lowercase(string(x))
    s == "infinity" && return Inf
    s == "-infinity" && return -Inf
    reject("bad number $x")
end

"""
Canonical JSON text of a value: keys sorted, no white space, all numbers as Float64.
The spec hash uses this text for the property. The kernel writes the same text.
"""
function canonical_json(x)
    if x isa AbstractDict
        keys_sorted = sort!(collect(string.(keys(x))))
        return "{" * join([JSON.json(k) * ":" * canonical_json(x[k]) for k in keys_sorted], ",") * "}"
    elseif x isa AbstractVector
        return "[" * join(canonical_json.(x), ",") * "]"
    elseif x isa AbstractFloat || x isa Integer
        return repr(Float64(x))            # the kernel stores property numbers as Float64
    else
        return JSON.json(x)
    end
end

"""Git tree hash of VeryDiff's src/ directory. A "-dirty" suffix shows local changes."""
function verydiff_src_tree()
    root = pkgdir(VeryDiff)
    tree = try strip(read(`git -C $root rev-parse HEAD:src`, String)) catch; "unknown" end
    dirty = try !isempty(strip(read(`git -C $root status --porcelain -- src`, String))) catch; true end
    return String(tree) * (dirty ? "-dirty" : "")
end

"""
SHA-256 over the two networks, the spec, the property and the VeryDiff source tree.
Each part is prefixed with its length, so that the parts cannot run into each other.
"""
function spec_hash(net1, net2, spec, property_text, tree)
    ctx = SHA.SHA256_CTX()
    for part in (read(net1), read(net2), read(spec), Vector{UInt8}(property_text), Vector{UInt8}(tree))
        SHA.update!(ctx, Vector{UInt8}(string(length(part), ":")))
        SHA.update!(ctx, part)
    end
    return bytes2hex(SHA.digest!(ctx))
end


# =====================================================================================================
# 3. Boxes
# =====================================================================================================

"""
Make a VeryDiff task for the box [lo, hi].

VeryDiff uses a centre and a radius per input dimension. Floating-point rounding can make this centre form
smaller than the box. To prevent this, widen the box by one ulp, round the radius up, and then use exact
rational arithmetic to make sure that the centre form contains the box.
"""
function task_for_box(lo::Vector{Float64}, hi::Vector{Float64}, active::Vector{Int}, cache::TaskBounds)
    mid = copy(lo)
    radius = Float64[]
    for i in active
        l = prevfloat(lo[i])
        h = nextfloat(hi[i])
        m = l / 2 + h / 2
        r = max(nextfloat(m - l), nextfloat(h - m))
        Q = Rational{BigInt}
        need(Q(m) - Q(r) <= Q(lo[i]) && Q(m) + Q(r) >= Q(hi[i]), "internal: centre form does not cover the box")
        mid[i] = m
        push!(radius, r)
    end
    return VerificationTask(mid, radius, copy(active), nothing, nothing, nothing, nothing, nothing, Inf, 1.0, cache)
end


# =====================================================================================================
# 4. Constraints
# =====================================================================================================
#
# A constraint is a fact about the pre-activations at the input of one ReLU layer. Lemmas and the
# constraints of hidden case splits use the same form. The variables are:
#   z1 = pre-activation in network 1,  z2 = pre-activation in network 2,  d = z1 − z2.

"""
A constraint at one ReLU layer:
  - interval bounds dlo ≤ d ≤ dhi for each unit in `neurons`,
  - optional interval bounds for z1 (`n1` = (lo, hi)) and z2 (`n2` = (lo, hi)),
  - optional linear relation Σ a·v ≤ rhs, with `terms` = [(variable, unit, a), ...].
Infinite bounds mean "no bound".
"""
struct Seed
    neurons::Vector{Int}
    dlo::Vector{Float64}
    dhi::Vector{Float64}
    n1::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}
    n2::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}
    terms::Vector{Tuple{String,Int,Float64}}
    rhs::Float64
end

const NO_TERMS = Tuple{String,Int,Float64}[]

"""
The constraint as a list of linear claims Σ a·v ≤ rhs. The order is part of the certificate format:
LP multipliers refer to the claims in this order. Infinite bounds give no claim.
"""
function linear_claims(s::Seed)
    claims = Tuple{Vector{Tuple{String,Int,Float64}},Float64}[]
    for (k, j) in enumerate(s.neurons)
        isfinite(s.dhi[k]) && push!(claims, ([("d", j, 1.0)], s.dhi[k]))
        isfinite(s.dlo[k]) && push!(claims, ([("d", j, -1.0)], -s.dlo[k]))
        if !isnothing(s.n1)
            isfinite(s.n1[2][k]) && push!(claims, ([("z1", j, 1.0)], s.n1[2][k]))
            isfinite(s.n1[1][k]) && push!(claims, ([("z1", j, -1.0)], -s.n1[1][k]))
        end
        if !isnothing(s.n2)
            isfinite(s.n2[2][k]) && push!(claims, ([("z2", j, 1.0)], s.n2[2][k]))
            isfinite(s.n2[1][k]) && push!(claims, ([("z2", j, -1.0)], -s.n2[1][k]))
        end
    end
    isempty(s.terms) || push!(claims, (s.terms, s.rhs))
    return claims
end

"""The constraint of one child of a hidden split "v_j ≤ t" (side 1) or "v_j ≥ t" (side 2)."""
function threshold_constraint(var::String, unit::Int, t::Float64, side::Int)
    lo, hi = side == 1 ? (-Inf, t) : (t, Inf)
    var == "d"  && return Seed([unit], [lo], [hi], nothing, nothing, NO_TERMS, Inf)
    var == "z1" && return Seed([unit], [-Inf], [Inf], ([lo], [hi]), nothing, NO_TERMS, Inf)
    return Seed([unit], [-Inf], [Inf], nothing, ([lo], [hi]), NO_TERMS, Inf)
end

"""The constraint of one child of a relational split "Σ a·v ≤ t" (side 1) or "Σ a·v ≥ t" (side 2)."""
function relational_constraint(terms, t::Float64, side::Int)
    side == 1 && return Seed(Int[], Float64[], Float64[], nothing, nothing, terms, t)
    return Seed(Int[], Float64[], Float64[], nothing, nothing, [(v, j, -a) for (v, j, a) in terms], -t)
end


# =====================================================================================================
# 5. Zonotope rows
# =====================================================================================================
#
# All zonotopes of one propagation use shared noise symbols ε ∈ [−1, 1]^m. The generator matrices are
# stored in blocks with an id. To compare rows of different zonotopes, map every (block id, column) to
# one position in a common vector of length m.

"""Common column layout of the zonotopes `Zs`: offset of each generator block, and total length m."""
function column_layout(Zs)
    width = Dict{Int,Int}()
    for Z in Zs, (block, id) in enumerate(collect(Z.generator_ids))
        width[id] = max(get(width, id, 0), size(Z.Gs[block], 2))
    end
    offset = Dict{Int,Int}()
    m = 0
    for id in sort(collect(keys(width)))
        offset[id] = m
        m += width[id]
    end
    return offset, m
end

"""Row `r` of zonotope `Z` as an affine function c + g·ε in the common layout."""
function zonotope_row(Z, r, offset, m)
    g = zeros(m)
    for (block, id) in enumerate(collect(Z.generator_ids)), col in 1:size(Z.Gs[block], 2)
        g[offset[id]+col] += Z.Gs[block][r, col]
    end
    return Z.c[r], g
end

"""Select z1, z2 or d from a differential zonotope."""
function select(DZ, var)
    var == "d" && return DZ.∂Z
    var == "z1" && return DZ.Z₁
    var == "z2" && return DZ.Z₂
    reject("bad variable name $var")
end

"""Σ a·v over the rows in `terms`, as an affine function c + g·ε."""
function combination_row(DZ, terms, offset, m)
    c = 0.0
    g = zeros(m)
    for (var, unit, a) in terms
        cc, gg = zonotope_row(select(DZ, var), unit, offset, m)
        c += a * cc
        g .+= a .* gg
    end
    return c, g
end


# =====================================================================================================
# 6. Bounds caches
# =====================================================================================================
#
# VeryDiff keeps, for each ReLU layer, a bounds cache: lower and upper bounds of z1, z2 and d. The ReLU
# transformer uses these bounds to choose its relaxation. Every bound in the cache must be sound for all
# concrete inputs of the node.

"""
Put constraint bounds into the cache of one ReLU layer with `n` units.
If the cache has no values yet, start from [−∞, +∞]. VeryDiff then intersects them with its own bounds.
"""
function seed_cache!(bc::BoundsCache, n::Int, seeds::Vector{Seed})
    if !bc.initialized
        bc.lower₁ = fill(-Inf, n); bc.upper₁ = fill(Inf, n)
        bc.lower₂ = fill(-Inf, n); bc.upper₂ = fill(Inf, n)
        bc.∂lower = fill(-Inf, n); bc.∂upper = fill(Inf, n)
        bc.initialized = true
    end
    for s in seeds
        need(all(1 .<= s.neurons .<= n), "constraint unit out of range")
        u = s.neurons
        bc.∂lower[u] .= max.(bc.∂lower[u], s.dlo)
        bc.∂upper[u] .= min.(bc.∂upper[u], s.dhi)
        if !isnothing(s.n1)
            bc.lower₁[u] .= max.(bc.lower₁[u], s.n1[1])
            bc.upper₁[u] .= min.(bc.upper₁[u], s.n1[2])
        end
        if !isnothing(s.n2)
            bc.lower₂[u] .= max.(bc.lower₂[u], s.n2[1])
            bc.upper₂[u] .= min.(bc.upper₂[u], s.n2[2])
        end
    end
end

"""Intersect the cache with the bounds of the input zonotope `Zi` (VeryDiff does the same in its transformer)."""
function intersect_with_zonotope!(bc::BoundsCache, Zi)
    b1 = zono_bounds(Zi.Z₁); b2 = zono_bounds(Zi.Z₂); bd = zono_bounds(Zi.∂Z)
    if !bc.initialized
        bc.lower₁ = b1[:, 1]; bc.upper₁ = b1[:, 2]
        bc.lower₂ = b2[:, 1]; bc.upper₂ = b2[:, 2]
        bc.∂lower = bd[:, 1]; bc.∂upper = bd[:, 2]
        bc.initialized = true
    else
        bc.lower₁ .= max.(bc.lower₁, b1[:, 1]); bc.upper₁ .= min.(bc.upper₁, b1[:, 2])
        bc.lower₂ .= max.(bc.lower₂, b2[:, 1]); bc.upper₂ .= min.(bc.upper₂, b2[:, 2])
        bc.∂lower .= max.(bc.∂lower, bd[:, 1]); bc.∂upper .= min.(bc.∂upper, bd[:, 2])
    end
end

"""
Tighten each bound with the identity z1 = z2 + d (one round, rounded outward).
Example: z1 ≥ lower(z2) + lower(d).
"""
function consistency_tighten!(bc::BoundsCache)
    bc.lower₁ .= max.(bc.lower₁, prevfloat.(bc.lower₂ .+ bc.∂lower))
    bc.upper₁ .= min.(bc.upper₁, nextfloat.(bc.upper₂ .+ bc.∂upper))
    bc.lower₂ .= max.(bc.lower₂, prevfloat.(bc.lower₁ .- bc.∂upper))
    bc.upper₂ .= min.(bc.upper₂, nextfloat.(bc.upper₁ .- bc.∂lower))
    bc.∂lower .= max.(bc.∂lower, prevfloat.(bc.lower₁ .- bc.upper₂))
    bc.∂upper .= min.(bc.∂upper, nextfloat.(bc.upper₁ .- bc.lower₂))
end

"""
True if some interval in the cache is empty by a clear margin. Then no concrete input satisfies the node's
constraints. The margin (relative 1e-9) protects against rounding. Infinite bounds never count as empty.
"""
function has_empty_interval(bc::BoundsCache)
    for (lower, upper) in ((bc.lower₁, bc.upper₁), (bc.lower₂, bc.upper₂), (bc.∂lower, bc.∂upper))
        for j in eachindex(lower)
            l, u = lower[j], upper[j]
            (isfinite(l) && isfinite(u)) || continue
            l - u > 1e-9 * (1 + max(abs(l), abs(u))) && return true
        end
    end
    return false
end

"""False if a cache contains NaN or an interval with lower > upper. Do not close a node in that case."""
function caches_are_valid(caches::TaskBounds)
    for bc in values(caches.bounds_cache)
        bc.initialized || continue
        any(isnan, vcat(bc.lower₁, bc.upper₁, bc.lower₂, bc.upper₂, bc.∂lower, bc.∂upper)) && return false
        (any(bc.lower₁ .> bc.upper₁) || any(bc.lower₂ .> bc.upper₂) || any(bc.∂lower .> bc.∂upper)) && return false
    end
    return true
end


# =====================================================================================================
# 7. Contraction and dual bounds
# =====================================================================================================
#
# A case-split constraint lo ≤ c + g·ε ≤ hi holds for every concrete input of the node. Two methods use it:
#   - Contraction: make the noise box [−1, 1]^m smaller, then rescale all live zonotopes to the new box.
#   - Dual bounds: for any λ ≥ 0, u = c_u + g_u·ε ≤ c_u + Σ|g_u − λ g| + λ (hi − c). Use the best λ.

"""
Make the box [lo, hi] smaller with the constraint g·ε ≤ rhs.
Return false if no point of the box satisfies the constraint.
"""
function contract_box!(lo, hi, g, rhs)
    isfinite(rhs) || return true
    r = rhs + 1e-12 * (abs(rhs) + sum(abs, g) + 1.0)            # allowance for rounding
    smallest = [x > 0 ? x * lo[i] : x * hi[i] for (i, x) in enumerate(g)]
    total = sum(smallest)
    total > r && return false
    for i in eachindex(g)
        x = g[i]
        abs(x) < 1e-300 && continue
        limit = (r - (total - smallest[i])) / x                 # bound on ε_i from all other terms at their minimum
        if x > 0
            limit = nextfloat(nextfloat(limit)) + 1e-12 * abs(limit)
            limit < hi[i] && (hi[i] = max(limit, lo[i]))
        else
            limit = prevfloat(prevfloat(limit)) - 1e-12 * abs(limit)
            limit > lo[i] && (lo[i] = min(limit, hi[i]))
        end
    end
    return true
end

"""All distinct zonotopes (Z₁, Z₂, ∂Z of every stored differential zonotope) in the propagation state."""
function live_zonotopes(P)
    Zs = Any[]
    seen = Set{UInt}()
    for entry in P.zono_storage.zonotopes
        (entry === nothing || entry.zonotope === nothing) && continue
        for Z in (entry.zonotope.Z₁, entry.zonotope.Z₂, entry.zonotope.∂Z)
            objectid(Z) in seen && continue
            push!(seen, objectid(Z))
            push!(Zs, Z)
        end
    end
    return Zs
end

"""
Contract the noise box with the constraints `seeds` at the input `Zi` of a ReLU layer, then rescale every live
zonotope: ε_i = mid_i + rad_i · ε′_i with ε′_i ∈ [−1, 1]. Return true if the constraints have no solution.
"""
function contract!(P, Zi, seeds::Vector{Seed})
    offset, m = column_layout([Zi.∂Z, Zi.Z₁, Zi.Z₂])
    lo = fill(-1.0, m)
    hi = fill(1.0, m)
    bounded_rows = Any[]                                        # (c, g, lower, upper) for lower ≤ c + g·ε ≤ upper
    for s in seeds, (k, j) in enumerate(s.neurons)
        isnothing(s.n1) || push!(bounded_rows, (zonotope_row(Zi.Z₁, j, offset, m)..., s.n1[1][k], s.n1[2][k]))
        isnothing(s.n2) || push!(bounded_rows, (zonotope_row(Zi.Z₂, j, offset, m)..., s.n2[1][k], s.n2[2][k]))
        if isfinite(s.dlo[k]) || isfinite(s.dhi[k])
            push!(bounded_rows, (zonotope_row(Zi.∂Z, j, offset, m)..., s.dlo[k], s.dhi[k]))
        end
    end
    for s in seeds
        isempty(s.terms) || push!(bounded_rows, (combination_row(Zi, s.terms, offset, m)..., -Inf, s.rhs))
    end
    for _ in 1:2, (c, g, lower, upper) in bounded_rows          # two rounds, as in the kernel
        contract_box!(lo, hi, g, upper - c) || return true
        contract_box!(lo, hi, -g, c - lower) || return true
    end
    any(lo .> hi) && return true

    # New centre and radius for each column that changed. The radius is rounded up.
    rescale = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
    for (id, start) in offset
        width = minimum([v for v in values(offset) if v > start]; init=m) - start
        for col in 1:width
            i = start + col
            lo[i] == -1.0 && hi[i] == 1.0 && continue
            mid = lo[i] / 2 + hi[i] / 2
            rescale[(id, col)] = (mid, nextfloat(max(hi[i] - mid, mid - lo[i])))
        end
    end
    isempty(rescale) && return false

    # First move the centres (with the old generators), then scale the generators.
    # A generator matrix can belong to more than one zonotope: scale each matrix only once.
    Zs = live_zonotopes(P)
    for Z in Zs, (block, id) in enumerate(collect(Z.generator_ids)), col in 1:size(Z.Gs[block], 2)
        haskey(rescale, (id, col)) && (Z.c .+= rescale[(id, col)][1] .* Z.Gs[block][:, col])
    end
    scaled = Set{UInt}()
    for Z in Zs, (block, id) in enumerate(collect(Z.generator_ids))
        G = Z.Gs[block]
        objectid(G) in scaled && continue
        push!(scaled, objectid(G))
        for col in 1:size(G, 2)
            haskey(rescale, (id, col)) && (G[:, col] .*= rescale[(id, col)][2])
        end
    end
    return false
end

"""
min over λ ≥ 0 of Σ|g − λa| + λβ. The function is convex and piecewise linear in λ. Its minimum is at one of
the break points g_i / a_i: walk them in increasing order until the slope is no longer negative.
"""
function best_dual_value(g, a, β)
    value_at_zero = sum(abs, g)
    isfinite(β) || return value_at_zero
    slope = β
    break_points = Tuple{Float64,Float64}[]
    for i in eachindex(a)
        a[i] == 0.0 && continue
        l = g[i] / a[i]
        if l > 0
            slope -= abs(a[i])
            push!(break_points, (l, abs(a[i])))
        else
            slope += abs(a[i])
        end
    end
    slope >= 0 && return value_at_zero
    sort!(break_points; by=first)
    λ = 0.0
    for (l, w) in break_points
        λ = l
        slope += 2w
        slope >= 0 && break
    end
    return min(value_at_zero, sum(abs, g .- λ .* a) + λ * β)
end

"""Find a layer by its index."""
layer_with_index(N, k) = only(l for l in get_layers(N) if l.layer_idx == k)

"""Input zonotope of ReLU layer k in the propagation state P."""
input_zonotope(P, N, k) = P.zono_storage.zonotopes[get_inputs(layer_with_index(N, k))[1]].zonotope

"""
Bound rows (option `bound_rows` = K > 0). A cached bound that is tighter than the zonotope's own bound is a true
row: for example z_j ≤ u_j, with z_j = c + g·ε. After the tightening of ReLU layer k, keep the K bounds with the
largest gap to the zonotope bound (relative to the width of the zonotope interval, only gaps > rtol). Ties go by
variable (z1, z2, d), then neuron, then upper before lower. The rows are used for the dual bounds of the later
ReLU layers. The kernel (VeryDiff `collect_bound_rows!`) uses the same rule. If a small difference in floating
point gives a different selection, the bounds can only change slightly; they are sound in all cases.

Each row is (layer, claim) with claim = ([(variable, neuron, ±1)], ±bound), as for linear_claims.
"""
function collect_bound_rows!(rows, bc, Zi, layer, K, rtol)
    K > 0 || return
    candidates = []
    for (name, Z, lower, upper) in (("z1", Zi.Z₁, bc.lower₁, bc.upper₁), ("z2", Zi.Z₂, bc.lower₂, bc.upper₂),
                                    ("d", Zi.∂Z, bc.∂lower, bc.∂upper))
        var = findfirst(==(name), ("z1", "z2", "d"))
        zb = zono_bounds(Z)
        for j in eachindex(lower)
            width = zb[j, 2] - zb[j, 1]
            width > 1e-12 || continue
            gap_upper = (zb[j, 2] - upper[j]) / width
            gap_lower = (lower[j] - zb[j, 1]) / width
            isfinite(upper[j]) && gap_upper > rtol && push!(candidates, (gap_upper, var, j, 0, ([(name, j, 1.0)], upper[j])))
            isfinite(lower[j]) && gap_lower > rtol && push!(candidates, (gap_lower, var, j, 1, ([(name, j, -1.0)], -lower[j])))
        end
    end
    sort!(candidates; by = x -> (-x[1], x[2], x[3], x[4]))
    for x in Iterators.take(candidates, K)
        push!(rows, (layer, x[5]))
    end
end

"""
Tighten the cache of ReLU layer k with the case-split constraints at layers ≤ k and the bound rows of layers < k
(dual bounds, one row at a time).
"""
function dual_tighten!(bc, P, N, Zi, k, constraints::Dict{Int,Vector{Seed}}, bound_rows=[])
    claims = [(layer, claim) for (layer, list) in constraints if layer <= k for s in list for claim in linear_claims(s)]
    append!(claims, [(layer, claim) for (layer, claim) in bound_rows if layer < k])
    isempty(claims) && return
    Zs = Any[Zi.∂Z, Zi.Z₁, Zi.Z₂]
    for layer in unique(first.(claims))
        Z = input_zonotope(P, N, layer)
        append!(Zs, [Z.∂Z, Z.Z₁, Z.Z₂])
    end
    offset, m = column_layout(Zs)
    rows = Tuple{Vector{Float64},Float64}[]                    # a·ε ≤ β
    for (layer, (terms, rhs)) in claims
        c, g = combination_row(input_zonotope(P, N, layer), terms, offset, m)
        push!(rows, (g, rhs - c))
    end
    for (Z, lower, upper) in ((Zi.Z₁, bc.lower₁, bc.upper₁), (Zi.Z₂, bc.lower₂, bc.upper₂), (Zi.∂Z, bc.∂lower, bc.∂upper))
        for j in eachindex(lower)
            c, g = zonotope_row(Z, j, offset, m)
            u = Inf
            l = -Inf
            for (a, β) in rows
                u = min(u, c + best_dual_value(g, a, β))
                l = max(l, c - best_dual_value(-g, a, β))
            end
            isfinite(u) && (upper[j] = min(upper[j], nextfloat(u + 1e-12 * abs(u))))
            isfinite(l) && (lower[j] = max(lower[j], prevfloat(l - 1e-12 * abs(l))))
        end
    end
end


# =====================================================================================================
# 8. Propagation of one node
# =====================================================================================================

"""
Propagate the box [lo, hi] through both networks.

  cache_in     the bounds cache that the node inherits (it is copied, not changed)
  seeds        constraints per ReLU layer that go into the cache (lemmas in scope and split constraints)
  constraints  split constraints per ReLU layer (for contraction and dual bounds)
  stop         nothing: propagate to the output; k: stop before ReLU layer k (lemma proofs)
  csb          apply the z1 = z2 + d tightening at every ReLU layer
  bound_rows   (K, rtol): collect K bound rows per ReLU layer for the dual bounds of later layers (K = 0: off)

At each ReLU layer, do these steps in this order: contraction, seeding, intersection with the zonotope,
dual bounds, consistency tightening, test for an empty interval, collection of bound rows, VeryDiff's
transformer. The kernel uses the same order.

Return a named tuple: Zin (input zonotope), Zout (output, or input of the stop layer), cache, valid
(no NaN and no inconsistent cache), P (the propagation state) and infeasible (the node has no concrete input).
"""
function propagate(N, lo, hi, active, cache_in::TaskBounds, seeds::Dict{Int,Vector{Seed}}, stop;
                   csb::Bool=false, constraints::Dict{Int,Vector{Seed}}=Dict{Int,Vector{Seed}}(),
                   bound_rows::Tuple{Int,Float64}=(0, 0.0))
    cache = deepcopy(cache_in)
    P = PropState(true)
    prepare_prop_state!(P, task_for_box(lo, hi, active, cache))
    layers = get_layers(N)
    max_position = maximum(max(maximum(l.inputs), maximum(l.outputs)) for l in layers)
    max_position > length(P.zono_storage.zonotopes) && resize_zonotope_storage!(P.zono_storage, max_position)
    infeasible() = (Zin=nothing, Zout=nothing, cache=cache, valid=false, P=P, infeasible=true)

    Zout = nothing
    collected = []
    for L in layers
        if !isnothing(stop) && L.layer_idx == stop
            Zout = P.zono_storage.zonotopes[get_inputs(L)[1]].zonotope
            break
        end
        inputs = get_zonos_at_pos(get_inputs(L), P)
        first_pass(P) && configure_first_usage!(L.layer_idx, inputs)
        zonos_initialized(P, get_outputs(L)) || init_layer!(P, L, inputs, get_outputs(L))
        input_zonotopes = get_zonotope.(inputs)
        if haskey(constraints, L.layer_idx) && contract!(P, input_zonotopes[1], constraints[L.layer_idx])
            return infeasible()
        end
        outputs = get_zonos_at_pos(get_outputs(L), P)
        bc = get!(() -> BoundsCache(), cache.bounds_cache, L.layer_idx)
        if haskey(seeds, L.layer_idx)
            need(L.layer1 isa ONNXRelu, "constraint on a layer that is not a ReLU layer")
            seed_cache!(bc, length(input_zonotopes[1].Z₁.c), seeds[L.layer_idx])
        end
        if L.layer1 isa ONNXRelu
            Zi = input_zonotopes[1]
            intersect_with_zonotope!(bc, Zi)
            dual_tighten!(bc, P, N, Zi, L.layer_idx, constraints, collected)
            (csb || haskey(seeds, L.layer_idx)) && consistency_tighten!(bc)
            has_empty_interval(bc) && return infeasible()
            collect_bound_rows!(collected, bc, Zi, L.layer_idx, bound_rows...)
        end
        propagate_layer!(outputs, L, input_zonotopes; bounds_cache=bc)
    end
    isnothing(stop) && (Zout = P.zono_storage.zonotopes[get_outputs(layers[end])[1]].zonotope)
    isnothing(Zout) && reject("stop layer $stop not found")

    valid = caches_are_valid(cache) && !any(Z -> any(isnan, zono_bounds(Z)), (Zout.∂Z, Zout.Z₁, Zout.Z₂))
    return (Zin=P.zono_storage.zonotopes[1].zonotope, Zout=Zout, cache=cache, valid=valid, P=P, infeasible=false)
end


# =====================================================================================================
# 9. LP evidence
# =====================================================================================================
#
# An LP leaf gives multipliers λ ≥ 0 for constraint rows A ε ≤ r. For an objective c + g·ε (e.g. one output
# difference), weak duality gives the bound  c + ‖g − Aᵀλ‖₁ + λ·r  for every ε in the box that satisfies
# the rows. The checker computes this bound itself. It does not solve an LP.
#
# The rows are, in this order: claims of the lemmas listed in the evidence, claims of the listed split
# constraints, and the listed relaxation rows.

# Relaxation rows: true relations for unit j at ReLU layer k (z = input, a = output, t = output of d).
const RELAXATION_TYPES = ("a1>=0", "a1>=z1", "a1<=ch", "a2>=0", "a2>=z2", "a2<=ch",
                          "dpre+", "dpre-", "dpost+", "dpost-", "t<=ch", "t>=ch")

"""
Slope s and left end l′ of the chord of ReLU over [min(l, 0), max(u, 0)]. The chord is
s·(z − l′). It lies above ReLU on the interval, for any sound bounds l ≤ z ≤ u.
"""
function chord(l, u)
    left = min(l, 0.0)
    right = max(u, 0.0)
    return right > left ? (right / (right - left), left) : (0.0, 0.0)
end

"""
One relaxation row as g·ε ≤ rhs. Zi and Zo are the input and output zonotopes of the layer, bc is its
cache. The table in certificate_format.md lists the relations.
"""
function relaxation_row(Zi, Zo, bc, j, type, offset, m)
    parts = Tuple{Float64,Any}[]              # Σ coefficient · zonotope row
    constant = 0.0
    if type == "a1>=0"                         # −a1 ≤ 0
        push!(parts, (-1.0, Zo.Z₁))
    elseif type == "a2>=0"
        push!(parts, (-1.0, Zo.Z₂))
    elseif type == "a1>=z1"                    # z1 − a1 ≤ 0
        push!(parts, (1.0, Zi.Z₁), (-1.0, Zo.Z₁))
    elseif type == "a2>=z2"
        push!(parts, (1.0, Zi.Z₂), (-1.0, Zo.Z₂))
    elseif type == "a1<=ch"                    # a1 − s·z1 + s·l′ ≤ 0
        s, left = chord(bc.lower₁[j], bc.upper₁[j])
        push!(parts, (1.0, Zo.Z₁), (-s, Zi.Z₁)); constant = s * left
    elseif type == "a2<=ch"
        s, left = chord(bc.lower₂[j], bc.upper₂[j])
        push!(parts, (1.0, Zo.Z₂), (-s, Zi.Z₂)); constant = s * left
    elseif type == "dpre+"                     # d − z1 + z2 ≤ 0 (with dpre−: d = z1 − z2)
        push!(parts, (1.0, Zi.∂Z), (-1.0, Zi.Z₁), (1.0, Zi.Z₂))
    elseif type == "dpre-"
        push!(parts, (-1.0, Zi.∂Z), (1.0, Zi.Z₁), (-1.0, Zi.Z₂))
    elseif type == "dpost+"                    # t − a1 + a2 ≤ 0 (with dpost−: t = a1 − a2)
        push!(parts, (1.0, Zo.∂Z), (-1.0, Zo.Z₁), (1.0, Zo.Z₂))
    elseif type == "dpost-"
        push!(parts, (-1.0, Zo.∂Z), (1.0, Zo.Z₁), (-1.0, Zo.Z₂))
    elseif type == "t<=ch"                     # t ≤ chord of max(0, d)
        s, left = chord(bc.∂lower[j], bc.∂upper[j])
        push!(parts, (1.0, Zo.∂Z), (-s, Zi.∂Z)); constant = s * left
    elseif type == "t>=ch"                     # t ≥ chord of min(0, d)
        s, left = chord(bc.∂lower[j], bc.∂upper[j])
        push!(parts, (1.0 - s, Zi.∂Z), (-1.0, Zo.∂Z)); constant = s * left
    else
        reject("unknown relaxation row type $type")
    end
    g = zeros(m)
    c = 0.0
    for (a, Z) in parts
        cc, gg = zonotope_row(Z, j, offset, m)
        g .+= a .* gg
        c += a * cc
    end
    rhs = -(c + constant)
    return g, rhs + 1e-12 * (1.0 + abs(rhs) + sum(abs, g))     # allowance for rounding
end

"""What a leaf's LP evidence may refer to."""
struct LPContext
    zonotope_at::Function        # k -> input zonotope of ReLU layer k (or nothing)
    lemmas::Dict{Int,Any}
    scope::Vector{Int}           # lemmas in scope at the leaf
    constraints::Vector{Tuple{Int,Seed}}
    relaxation::Union{Nothing,NamedTuple}   # (input, output, cache, usable) if relaxation rows are allowed
end

"""Build the constraint rows A ε ≤ r named by the evidence `ev`. Reject invalid references."""
function evidence_rows(ev, ctx::LPContext, offset, m)
    rows = Vector{Float64}[]
    rhs = Float64[]
    add_claims!(Zk, s) = for (terms, b) in linear_claims(s)
        c, g = combination_row(Zk, terms, offset, m)
        push!(rows, g); push!(rhs, b - c)
    end

    used_lemmas = Int.(ev["lemmas"])
    need(issubset(used_lemmas, ctx.scope), "LP evidence uses lemmas that are not in scope")
    need(issorted(used_lemmas) && allunique(used_lemmas), "LP evidence lemma list is not sorted and unique")
    for L in used_lemmas
        Zk = ctx.zonotope_at(ctx.lemmas[L].layer)
        need(!isnothing(Zk), "LP evidence uses lemma $L, but its layer is not available")
        add_claims!(Zk, ctx.lemmas[L].seed)
    end

    used_constraints = Int.(get(ev, "cons", Int[]))
    need(issorted(used_constraints) && allunique(used_constraints) && all(1 .<= used_constraints .<= length(ctx.constraints)),
         "LP evidence constraint list is not valid")
    for i in used_constraints
        layer, s = ctx.constraints[i]
        Zk = ctx.zonotope_at(layer)
        need(!isnothing(Zk), "LP evidence uses constraint $i, but its layer is not available")
        add_claims!(Zk, s)
    end

    if haskey(ev, "relax")
        rx = ctx.relaxation
        need(!isnothing(rx), "LP evidence uses relaxation rows, but the certificate does not enable them")
        for entry in ev["relax"]
            k = Int(entry[1]); j = Int(entry[2]); type = String(entry[3])
            need(type in RELAXATION_TYPES, "unknown relaxation row type $type")
            need(rx.usable(k), "relaxation row on layer $k, which this propagation did not compute")
            bc = rx.cache.bounds_cache[k]
            need(1 <= j <= length(bc.lower₁), "relaxation row unit out of range")
            g, b = relaxation_row(rx.input(k), rx.output(k), bc, j, type, offset, m)
            push!(rows, g); push!(rhs, b)
        end
    end
    A = isempty(rows) ? zeros(0, m) : permutedims(reduce(hcat, rows))
    return A, rhs
end

"""
Check LP evidence. `objectives` maps a label to (c, g, threshold): each objective c + g·ε must be at most
its threshold. An objective with multipliers in the evidence uses the dual bound. An objective without
multipliers must hold with the plain zonotope bound c + ‖g‖₁.
"""
function lp_evidence_holds(ev, objectives, ctx::LPContext, offset, m)
    A, r = evidence_rows(ev, ctx, offset, m)
    multipliers = Dict(String(e["o"]) => Float64.(e["lam"]) for e in ev["rows"])
    for (label, (c, g, threshold)) in objectives
        if haskey(multipliers, label)
            λ = multipliers[label]
            need(length(λ) == size(A, 1), "LP evidence for $label has the wrong number of multipliers")
            need(all(>=(0.0), λ) && all(isfinite, λ), "LP evidence for $label has negative or non-finite multipliers")
            bound = c + sum(abs, g .- A' * λ) + (isempty(λ) ? 0.0 : sum(λ .* r))
        else
            bound = c + sum(abs, g)
        end
        bound <= threshold || return false
    end
    return true
end


# =====================================================================================================
# 10. Reading the certificate
# =====================================================================================================

"""The verification problem, read from the files (not from the certificate)."""
struct Problem
    N::Any                       # differential network (VeryDiff GeminiNetwork)
    onnx1::Any
    onnx2::Any
    exec1::Any                   # executable networks (used by VeryDiff's epsilon property)
    exec2::Any
    lo::Vector{Float64}          # input box
    hi::Vector{Float64}
    active::Vector{Int}          # input dimensions with lo < hi
    relu_layers::Set{Int}
    output_tensor::String
    softmax_removed::Bool
    epsilon::Float64
end

"""One proof-tree node as written in the certificate."""
struct CNode
    id::Int
    parent::Int
    side::Int                    # 0: root or child of an assume node; 1: lower child; 2: upper child
    kind::String                 # "s" split, "a" assume, "c" closed, "o" open
    payload::Dict{String,Any}
    region::Vector{Any}          # [lo, hi] on the active dimensions (redundant; must match the replay)
end

"""The proof tree, the obligations and the lemmas of the certificate."""
struct Tree
    nodes::Dict{Int,CNode}
    children::Dict{Int,Vector{Int}}
    obligations::Dict{Int,NamedTuple}
    lemmas::Dict{Int,NamedTuple}
    main::NamedTuple
end

function load_json(path)
    text = endswith(path, ".gz") ? String(transcode(GzipDecompressor, read(path))) : read(path, String)
    return JSON.parse(text)
end

"""Read the networks and the spec. Compare them with the header of the certificate."""
function load_problem(C, net1, net2, spec)
    need(get(C, "version", 0) == 1, "unsupported certificate version")
    property = C["property"]
    need(get(property, "kind", "") == "epsilon", "the checker supports only epsilon properties")
    epsilon = Float64(property["epsilon"])
    need(isfinite(epsilon) && epsilon >= 0, "bad epsilon")
    tree = verydiff_src_tree()
    need(C["verydiff_src_tree"] == tree, "VeryDiff src tree mismatch (certificate $(C["verydiff_src_tree"]), local $tree)")
    property_text = canonical_json(Dict("kind" => "epsilon", "epsilon" => epsilon))
    need(spec_hash(net1, net2, spec, property_text, tree) == C["spec_hash"], "spec hash mismatch")

    onnx1 = load_onnx_model(net1)
    onnx2 = load_onnx_model(net2)
    ast, n_inputs, _ = get_ast(spec)
    boxes = Matrix{Float64}[]
    for (bounds, _, _, _) in ast              # the iterator has no length, so use a loop
        push!(boxes, Matrix{Float64}(bounds[1:n_inputs, :]))
    end
    need(length(boxes) == 1, "the spec must have exactly one input box")
    lo = boxes[1][:, 1]
    hi = boxes[1][:, 2]
    need(Float64.(C["spec_box"]["lo"]) == lo && Float64.(C["spec_box"]["hi"]) == hi, "spec box mismatch")
    active = findall(lo .< hi)
    need(Int.(C["active_dims"]) == active, "active dims mismatch")

    N = GeminiNetwork(onnx1, onnx2)
    _, io_map = sort_network(onnx1)
    tensor_name = Dict(v => string(k) for (k, v) in io_map)
    softmax_removed = N.diff_layers[end].layer1 isa ONNXSoftmax
    softmax_removed && pop!(N.diff_layers)    # epsilon refers to the outputs before the Softmax
    need(softmax_removed == C["softmax_removed"], "softmax_removed mismatch")
    output_tensor = tensor_name[get_outputs(N.diff_layers[end])[1]]
    relu_layers = Set(l.layer_idx for l in N.diff_layers if l.layer1 isa ONNXRelu)
    return Problem(N, onnx1, onnx2, executable_network(onnx1), executable_network(onnx2), lo, hi, active,
                   relu_layers, output_tensor, softmax_removed, epsilon)
end

"""Read a lemma entry and make its constraint."""
function read_lemma(l, problem::Problem)
    id = Int(l["id"])
    n1 = haskey(l, "n1") ? (decode_number.(l["n1"][1]), decode_number.(l["n1"][2])) : nothing
    n2 = haskey(l, "n2") ? (decode_number.(l["n2"][1]), decode_number.(l["n2"][2])) : nothing
    terms = haskey(l, "terms") ? [(String(t[1]), Int(t[2]), Float64(t[3])) for t in l["terms"]] : NO_TERMS
    seed = Seed(Int.(l["neurons"]), decode_number.(l["dlo"]), decode_number.(l["dhi"]), n1, n2, terms,
                haskey(l, "rhs") ? Float64(l["rhs"]) : Inf)
    k = length(seed.neurons)
    need(length(seed.dlo) == k && length(seed.dhi) == k, "lemma $id is malformed")
    need(Int(l["relu_layer"]) in problem.relu_layers, "lemma $id is on a layer that is not a ReLU layer")
    return (id=id, obligation=Int(l["obligation"]), scope=Int(l["scope_node"]), layer=Int(l["relu_layer"]),
            seed=seed, claimed_proven=(get(l, "status", "") == "proven"))
end

"""Read the tree and check its shape: parents, children per node kind, obligations and lemma references."""
function read_tree(C, problem::Problem)
    nodes = Dict{Int,CNode}()
    for e in C["nodes"]
        id = Int(e[1])
        need(!haskey(nodes, id), "duplicate node id $id")
        nodes[id] = CNode(id, Int(e[2]), Int(e[3]), String(e[4]), Dict{String,Any}(e[5]), e[6])
    end
    children = Dict{Int,Vector{Int}}(id => Int[] for id in keys(nodes))
    for (id, n) in nodes
        n.parent == 0 && continue
        need(haskey(nodes, n.parent), "node $id has a missing parent $(n.parent)")
        push!(children[n.parent], id)
    end

    obligations = Dict{Int,NamedTuple}()
    for o in C["obligations"]
        id = Int(o["id"])
        obligations[id] = (id=id, kind=String(o["kind"]), root=Int(o["root"]), target=String(o["target"]))
    end
    mains = [o for o in values(obligations) if o.kind == "main"]
    need(length(mains) == 1, "the certificate needs exactly one main obligation")

    lemmas = Dict{Int,NamedTuple}()
    for l in C["lemmas"]
        lemma = read_lemma(l, problem)
        lemmas[lemma.id] = lemma
        o = get(obligations, lemma.obligation, nothing)
        need(!isnothing(o) && o.kind == "lemma" && o.target == "lemma:$(lemma.id)",
             "lemma $(lemma.id) has no matching obligation")
    end

    for o in values(obligations)
        need(haskey(nodes, o.root) && nodes[o.root].parent == 0 && nodes[o.root].side == 0,
             "obligation $(o.id) has an invalid root")
        o.kind == "lemma" && need(any(l.obligation == o.id for l in values(lemmas)), "lemma obligation $(o.id) has no lemma")
    end
    roots = Set(o.root for o in values(obligations))
    for (id, n) in nodes
        n.parent == 0 && need(id in roots, "orphan node $id")
        ch = children[id]
        if n.kind == "s"
            need(length(ch) == 2 && Set(nodes[c].side for c in ch) == Set([1, 2]), "split node $id needs a lower and an upper child")
            need(n.payload["b"] in ("primary", "relu", "lin"), "split type $(n.payload["b"]) is not supported")
        elseif n.kind == "a"
            need(length(ch) == 1 && nodes[ch[1]].side == 0, "assume node $id needs exactly one child")
            L = Int(n.payload["l"])
            need(haskey(lemmas, L) && lemmas[L].scope == id, "assume node $id refers to an unknown lemma $L")
        elseif n.kind in ("c", "o")
            need(isempty(ch), "leaf $id has children")
        else
            reject("unknown node kind $(n.kind)")
        end
    end
    for l in values(lemmas)
        scope = get(nodes, l.scope, nothing)
        need(!isnothing(scope) && scope.kind == "a" && Int(scope.payload["l"]) == l.id, "lemma $(l.id) has an invalid scope node")
    end
    return Tree(nodes, children, obligations, lemmas, mains[1])
end


# =====================================================================================================
# 11. Replay of the proof tree
# =====================================================================================================
#
# Walk each obligation's tree from the root. Every node gets its region, its inherited cache, the lemmas
# in scope and the split constraints from the path. A leaf gets a verdict (`verified`). An inner node passes
# these values to its children. If a split or assume node is marked "p" (propagated), its children inherit
# the cache after one propagation of that node; else they inherit the node's own inherited cache.

"""State of one node during the replay."""
struct NodeState
    id::Int
    cache::TaskBounds
    scope::Vector{Int}                       # lemmas in scope
    constraints::Vector{Tuple{Int,Seed}}     # (ReLU layer, constraint) from the hidden splits on the path
end

mutable struct Replay
    problem::Problem
    tree::Tree
    csb::Bool
    lp_relax::Bool
    bound_rows::Tuple{Int,Float64}           # options bound_rows, bound_rows_rtol
    epsilon_property::Function
    region::Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}
    lemma_start::Dict{Int,NodeState}         # lemma root id -> state at its assume node (lemma chaining)
    obligation_of::Dict{Int,Int}
    verified::Dict{Int,Bool}                 # leaf id -> evidence holds
    verbose::Bool
end

"""Seeds per ReLU layer (lemmas in scope and split constraints), and split constraints per ReLU layer."""
function seeds_and_constraints(r::Replay, s::NodeState)
    seeds = Dict{Int,Vector{Seed}}()
    for L in s.scope
        push!(get!(seeds, r.tree.lemmas[L].layer, Seed[]), r.tree.lemmas[L].seed)
    end
    constraints = Dict{Int,Vector{Seed}}()
    for (k, c) in s.constraints
        push!(get!(seeds, k, Seed[]), c)
        push!(get!(constraints, k, Seed[]), c)
    end
    return seeds, constraints
end

"""Propagate the node of state `s` (region from the replay)."""
function propagate_node(r::Replay, s::NodeState, stop)
    lo, hi = r.region[s.id]
    seeds, constraints = seeds_and_constraints(r, s)
    return propagate(r.problem.N, lo, hi, r.problem.active, s.cache, seeds, stop; csb=r.csb, constraints=constraints,
                     bound_rows=r.bound_rows)
end

"""Lemma of a lemma obligation (nothing for the main obligation)."""
target_lemma(r::Replay, o) = o.kind == "main" ? nothing : r.tree.lemmas[parse(Int, split(o.target, ":")[2])]

"""
Check a leaf with LP evidence. The objectives are ±(N1 − N2)_i ≤ ε for the main obligation, or the lemma's
claims for a lemma obligation.
"""
function check_lp_leaf(r::Replay, o, s::NodeState, prop, ev, stop)
    N = r.problem.N
    Z = prop.Zout
    P = prop.P
    zonotope_at(k) = begin
        !isnothing(stop) && k > stop && return nothing
        !isnothing(stop) && k == stop && return Z
        entry = P.zono_storage.zonotopes[get_inputs(layer_with_index(N, k))[1]]
        isnothing(entry) ? nothing : entry.zonotope
    end
    zonotope_of(k, side) = P.zono_storage.zonotopes[side(layer_with_index(N, k))[1]].zonotope
    usable(k) = k in r.problem.relu_layers && (isnothing(stop) || k < stop) && haskey(prop.cache.bounds_cache, k) &&
                !isnothing(P.zono_storage.zonotopes[get_outputs(layer_with_index(N, k))[1]])

    # Column layout over all zonotopes that the objectives and the rows use.
    Zs = Any[Z.∂Z, Z.Z₁, Z.Z₂]
    layers_used = vcat([r.tree.lemmas[L].layer for L in Int.(ev["lemmas"])],
                       [s.constraints[i][1] for i in Int.(get(ev, "cons", Int[])) if 1 <= i <= length(s.constraints)])
    for k in layers_used
        Zk = zonotope_at(k)
        isnothing(Zk) || append!(Zs, [Zk.∂Z, Zk.Z₁, Zk.Z₂])
    end
    if haskey(ev, "relax")
        need(r.lp_relax, "the certificate uses relaxation rows without the lp_relax option")
        for k in sort(collect(r.problem.relu_layers))
            usable(k) || continue
            for side in (get_inputs, get_outputs)
                ZZ = zonotope_of(k, side)
                append!(Zs, [ZZ.∂Z, ZZ.Z₁, ZZ.Z₂])
            end
        end
    end
    offset, m = column_layout(Zs)

    objectives = Dict{String,Tuple{Float64,Vector{Float64},Float64}}()
    lemma = target_lemma(r, o)
    if isnothing(lemma)
        for i in 1:length(Z.∂Z.c)
            c, g = zonotope_row(Z.∂Z, i, offset, m)
            objectives["out:$i:+"] = (c, g, r.problem.epsilon)
            objectives["out:$i:-"] = (-c, -g, r.problem.epsilon)
        end
    else
        for (k, (terms, rhs)) in enumerate(linear_claims(lemma.seed))
            c, g = combination_row(Z, terms, offset, m)
            objectives["spec:$k"] = (c, g, rhs)
        end
    end
    relaxation = haskey(ev, "relax") ?
        (input=k -> zonotope_of(k, get_inputs), output=k -> zonotope_of(k, get_outputs), cache=prop.cache, usable=usable) : nothing
    ctx = LPContext(zonotope_at, r.tree.lemmas, s.scope, s.constraints, relaxation)
    return prop.valid && lp_evidence_holds(ev, objectives, ctx, offset, m)
end

"""Check a lemma leaf without LP evidence: the lemma's bounds must hold at the stop layer."""
function check_lemma_leaf(lemma, prop)
    s = lemma.seed
    Z = prop.Zout
    bd = zono_bounds(Z.∂Z)
    holds = all(bd[s.neurons, 1] .>= s.dlo) && all(bd[s.neurons, 2] .<= s.dhi)
    if !isnothing(s.n1)
        b1 = zono_bounds(Z.Z₁)
        holds &= all(b1[s.neurons, 1] .>= s.n1[1]) && all(b1[s.neurons, 2] .<= s.n1[2])
    end
    if !isnothing(s.n2)
        b2 = zono_bounds(Z.Z₂)
        holds &= all(b2[s.neurons, 1] .>= s.n2[1]) && all(b2[s.neurons, 2] .<= s.n2[2])
    end
    if !isempty(s.terms)
        offset, m = column_layout([Z.∂Z, Z.Z₁, Z.Z₂])
        c, g = combination_row(Z, s.terms, offset, m)
        holds &= c + sum(abs, g) <= s.rhs
    end
    return holds && prop.valid
end

"""Give a verdict for a closed leaf."""
function check_closed_leaf(r::Replay, o, s::NodeState, n::CNode)
    lemma = target_lemma(r, o)
    stop = isnothing(lemma) ? nothing : lemma.layer
    prop = propagate_node(r, s, stop)
    if get(n.payload, "inf", false) == true
        return prop.infeasible                      # the node claims: no concrete input satisfies its constraints
    end
    prop.infeasible && return false                 # a normal leaf needs a normal propagation
    ev = get(n.payload, "lp", nothing)
    !isnothing(ev) && return check_lp_leaf(r, o, s, prop, ev, stop)
    if isnothing(lemma)
        sat, _, _, _, distance_bound = r.epsilon_property(r.problem.exec1, r.problem.exec2, prop.Zin, prop.Zout, nothing)
        return sat && prop.valid && !isnan(distance_bound)
    end
    return check_lemma_leaf(lemma, prop)
end

"""Children of an inner node, each with its region and state."""
function expand(r::Replay, s::NodeState, n::CNode, out_cache::TaskBounds)
    lo, hi = r.region[n.id]
    nodes = r.tree.nodes
    kids = Tuple{Int,Tuple{Vector{Float64},Vector{Float64}},NodeState}[]
    if n.kind == "s" && n.payload["b"] == "relu"
        # Hidden split "v_j ≤ t" / "v_j ≥ t" at the input of ReLU layer k. The region does not change.
        k = Int(n.payload["k"]); var = String(n.payload["v"]); j = Int(n.payload["j"]); t = Float64(n.payload["c"])
        need(k in r.problem.relu_layers, "hidden split node $(n.id) is on layer $k, which is not a ReLU layer")
        need(var in ("z1", "z2", "d") && isfinite(t) && j >= 1, "hidden split node $(n.id) is malformed")
        for c in r.tree.children[n.id]
            constraint = threshold_constraint(var, j, t, nodes[c].side)
            push!(kids, (c, (copy(lo), copy(hi)), NodeState(c, deepcopy(out_cache), copy(s.scope), vcat(s.constraints, [(k, constraint)]))))
        end
    elseif n.kind == "s" && n.payload["b"] == "lin"
        # Relational split "Σ a·v ≤ t" / "Σ a·v ≥ t" at the input of ReLU layer k. The region does not change.
        k = Int(n.payload["k"]); t = Float64(n.payload["c"])
        need(k in r.problem.relu_layers && isfinite(t), "relational split node $(n.id) is malformed")
        terms = [(String(x[1]), Int(x[2]), Float64(x[3])) for x in n.payload["terms"]]
        need(!isempty(terms) && all(x -> x[1] in ("z1", "z2", "d") && x[2] >= 1 && isfinite(x[3]), terms),
             "relational split node $(n.id) has bad terms")
        for c in r.tree.children[n.id]
            constraint = relational_constraint(terms, t, nodes[c].side)
            push!(kids, (c, (copy(lo), copy(hi)), NodeState(c, deepcopy(out_cache), copy(s.scope), vcat(s.constraints, [(k, constraint)]))))
        end
    elseif n.kind == "s"
        # Input split: child 1 gets [lo, cut], child 2 gets [cut, hi] in dimension d.
        d = Int(n.payload["d"]); cut = Float64(n.payload["c"])
        need(d in r.problem.active, "split node $(n.id) is on an inactive dimension $d")
        need(lo[d] < cut < hi[d], "split node $(n.id): cut $cut is not strictly inside ($(lo[d]), $(hi[d]))")
        for c in r.tree.children[n.id]
            l2 = copy(lo); h2 = copy(hi)
            nodes[c].side == 1 ? (h2[d] = cut) : (l2[d] = cut)
            push!(kids, (c, (l2, h2), NodeState(c, deepcopy(out_cache), copy(s.scope), copy(s.constraints))))
        end
    else
        # Assume node: the child has lemma L in scope. The lemma's own proof starts from this node's state.
        L = Int(n.payload["l"])
        need(!(L in s.scope), "lemma $L is in scope of its own assume node")
        c = r.tree.children[n.id][1]
        push!(kids, (c, (copy(lo), copy(hi)), NodeState(c, deepcopy(out_cache), vcat(s.scope, [L]), copy(s.constraints))))
        lemma_root = r.tree.obligations[r.tree.lemmas[L].obligation].root
        r.region[lemma_root] = (copy(lo), copy(hi))
        r.lemma_start[lemma_root] = NodeState(lemma_root, deepcopy(out_cache), copy(s.scope), copy(s.constraints))
    end
    return kids
end

"""Replay the tree of one obligation."""
function replay_obligation!(r::Replay, o)
    start = o.kind == "main" ? NodeState(o.root, TaskBounds(), Int[], Tuple{Int,Seed}[]) : r.lemma_start[o.root]
    lemma = target_lemma(r, o)
    stop = isnothing(lemma) ? nothing : lemma.layer
    stack = [NodeState(o.root, deepcopy(start.cache), copy(start.scope), copy(start.constraints))]
    while !isempty(stack)
        s = pop!(stack)
        need(!haskey(r.obligation_of, s.id), "node $(s.id) is reached twice")
        r.obligation_of[s.id] = o.id
        n = r.tree.nodes[s.id]
        lo, hi = r.region[s.id]
        need(Float64.(n.region[1]) == lo[r.problem.active] && Float64.(n.region[2]) == hi[r.problem.active],
             "the stored region of node $(s.id) differs from the reconstruction")
        if n.kind == "c"
            r.verified[s.id] = check_closed_leaf(r, o, s, n)
            r.verbose && println(stderr, "leaf $(s.id): ", r.verified[s.id])
            continue
        elseif n.kind == "o"
            r.verified[s.id] = false
            continue
        end
        out_cache = n.payload["p"] == true ? propagate_node(r, s, stop).cache : s.cache
        for (child, child_region, child_state) in expand(r, s, n, out_cache)
            r.region[child] = child_region
            push!(stack, child_state)
        end
    end
end

"""
Replay all obligations. Do the main obligation first. A lemma obligation can start when the replay has
reached its assume node (this gives its region and its start state).
"""
function replay_all!(r::Replay)
    t = r.tree
    r.region[t.main.root] = (copy(r.problem.lo), copy(r.problem.hi))
    replay_obligation!(r, t.main)
    done = Set([t.main.id])
    progress = true
    while progress
        progress = false
        for o in values(t.obligations)
            (o.id in done || !haskey(r.region, o.root)) && continue
            replay_obligation!(r, o)
            push!(done, o.id)
            progress = true
        end
    end
    need(length(done) == length(t.obligations), "some obligations are unreachable (their assume node is not in the tree)")
    need(length(r.obligation_of) == length(t.nodes), "some nodes are not reachable from an obligation root")
end


# =====================================================================================================
# 12. Closure and result
# =====================================================================================================
#
# A closed leaf is closed if its evidence holds. A split node is closed if both children are closed. An assume
# node is closed if its child is closed and its lemma is proven. A lemma is proven if its proof tree is
# closed and all lemmas that it inherits are proven. A cycle of lemmas is an error.

mutable struct Closure
    r::Replay
    node_memo::Dict{Int,Bool}
    lemma_memo::Dict{Int,Bool}
    in_progress::Set{Int}
end
Closure(r::Replay) = Closure(r, Dict{Int,Bool}(), Dict{Int,Bool}(), Set{Int}())

function is_closed(cl::Closure, id::Int)
    haskey(cl.node_memo, id) && return cl.node_memo[id]
    n = cl.r.tree.nodes[id]
    children = cl.r.tree.children[id]
    closed = n.kind == "c" ? cl.r.verified[id] :
             n.kind == "o" ? false :
             n.kind == "s" ? all(c -> is_closed(cl, c), children) :
             (is_closed(cl, children[1]) && is_proven(cl, Int(n.payload["l"])))
    cl.node_memo[id] = closed
    return closed
end

function is_proven(cl::Closure, L::Int)
    haskey(cl.lemma_memo, L) && return cl.lemma_memo[L]
    need(!(L in cl.in_progress), "circular lemma dependency at lemma $L")
    push!(cl.in_progress, L)
    root = cl.r.tree.obligations[cl.r.tree.lemmas[L].obligation].root
    proven = is_closed(cl, root) && all(l -> is_proven(cl, l), cl.r.lemma_start[root].scope)
    delete!(cl.in_progress, L)
    cl.lemma_memo[L] = proven
    return proven
end

"""
Volume fraction of the input box that closed subtrees of the main tree cover (for PARTIAL certificates).
An input split divides the volume in proportion to the cut. A hidden or relational split does not divide the
region: its children cover parts of the same region of unknown size, so count it only if it is closed as a whole.
"""
function closed_volume(cl::Closure)
    t = cl.r.tree
    total = 0.0
    stack = [(t.main.root, 1.0)]
    while !isempty(stack)
        id, volume = pop!(stack)
        if is_closed(cl, id)
            total += volume
            continue
        end
        n = t.nodes[id]
        (n.kind == "s" && n.payload["b"] == "primary") || continue
        d = Int(n.payload["d"]); cut = Float64(n.payload["c"])
        lo, hi = cl.r.region[id]
        lower_part = (cut - lo[d]) / (hi[d] - lo[d])
        for c in t.children[id]
            push!(stack, (c, volume * (t.nodes[c].side == 1 ? lower_part : 1 - lower_part)))
        end
    end
    return total
end

"""Check a certificate. Return the ACCEPT text, or throw Reject."""
function check(cert_path, net1, net2, spec; verbose=false)
    VeryDiff.USE_DIFFZONO[] = true
    VeryDiff.NEW_HEURISTIC[] = true
    C = load_json(cert_path)
    problem = load_problem(C, net1, net2, spec)

    status = C["status"]
    need(status in ("SAFE", "UNSAFE", "PARTIAL"), "bad status")
    status == "UNSAFE" && return check_counterexample(C, problem)
    status == "SAFE" && need(isnothing(C["cex"]), "a SAFE certificate must not have a counterexample")

    options = get(C, "options", Dict())
    VeryDiff.ONE_SIDED_DIFF_FALLBACK[] = Bool(get(options, "fallback", false))
    tree = read_tree(C, problem)
    bound_rows = (Int(get(options, "bound_rows", 0)), Float64(get(options, "bound_rows_rtol", 0.05)))
    need(bound_rows[1] >= 0 && bound_rows[2] >= 0, "bad bound_rows options")
    r = Replay(problem, tree, Bool(get(options, "csb", false)), Bool(get(options, "lp_relax", false)), bound_rows,
               VeryDiff.get_epsilon_property(problem.epsilon), Dict(), Dict(), Dict(), Dict(), verbose)
    replay_all!(r)

    cl = Closure(r)
    for l in values(tree.lemmas)
        l.claimed_proven && need(is_proven(cl, l.id), "lemma $(l.id) is claimed proven, but its proof does not replay")
    end
    failed = [id for (id, n) in tree.nodes if n.kind == "c" && !r.verified[id]]
    need(isempty(failed), "closed leaves that do not check: $(failed[1:min(5, end)])")
    if status == "SAFE"
        need(is_closed(cl, tree.main.root), "the main obligation is not closed")
        return "ACCEPT SAFE nodes=$(length(tree.nodes)) lemmas=$(length(tree.lemmas))"
    end
    return "ACCEPT PARTIAL verified_closed_volume=$(round(closed_volume(cl); digits=6))"
end

"""UNSAFE: evaluate both networks at the counterexample. The tree is not used."""
function check_counterexample(C, problem::Problem)
    cex = C["cex"]
    need(!isnothing(cex), "an UNSAFE certificate needs a counterexample")
    x = Float64.(cex["x"])
    need(length(x) == length(problem.lo) && all(problem.lo .<= x .<= problem.hi), "the counterexample is outside the spec box")
    y1 = Float64.(vec(compute_all_outputs(problem.onnx1, x)[problem.output_tensor]))
    y2 = Float64.(vec(compute_all_outputs(problem.onnx2, x)[problem.output_tensor]))
    distance = maximum(abs.(y1 .- y2))
    need(distance > problem.epsilon, "the counterexample does not violate the property (distance $distance <= epsilon $(problem.epsilon))")
    return "ACCEPT UNSAFE dist=$(distance) eps=$(problem.epsilon)"
end


# =====================================================================================================
# 13. Command line
# =====================================================================================================

function main(args)
    if length(args) < 4
        println(stderr, "usage: check_certificate.jl CERT NET1 NET2 SPEC [--verbose]")
        return 2
    end
    try
        println(check(args[1], args[2], args[3], args[4]; verbose="--verbose" in args))
        return 0
    catch e
        if e isa Reject
            println("REJECT ", e.msg)
        else
            println("REJECT checker error: ", sprint(showerror, e)[1:min(end, 500)])
        end
        return 1
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(CertChecker.main(ARGS))
end
