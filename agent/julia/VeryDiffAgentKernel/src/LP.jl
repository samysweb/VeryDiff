# LP tightening with lemma constraints, certified by weak duality.
#
# All zonotopes of one propagation are affine functions of shared noise symbols ε ∈ [-1,1]^m
# (generator blocks are identified by id; owned blocks only append columns, so column c of block g denotes
# the same symbol at every layer). For each concrete execution there is one ε that reproduces it at every
# layer. A lemma proven for the node's region is therefore a valid linear constraint A ε ≤ r on the ε of
# concrete executions, and
#     max_ε  c + g·ε   s.t.  A ε ≤ r, ε ∈ [-1,1]^m
# bounds the objective. For ANY λ ≥ 0 (weak duality):
#     c + ‖g − Aᵀλ‖₁ + λ·r   ≥   that maximum.
# GLPK only proposes λ; the bound is recomputed here (and by the checker) from λ, so solver errors can only
# make the bound looser, never unsound (same Float64 trust level as zono_bounds).

using JuMP
using GLPK

const LP_TIME_LIMIT = Ref{Float64}(2.0)

struct AffRow
    c::Float64
    g::Vector{Float64}
end

"""Variable layout over the generator blocks of the given zonotopes: id => column offset, total m."""
function var_layout(Zs)
    sizes = Dict{Int,Int}()
    for Z in Zs
        for (bi, id) in enumerate(collect(Z.generator_ids))
            sizes[id] = max(get(sizes, id, 0), size(Z.Gs[bi], 2))
        end
    end
    off = Dict{Int,Int}(); o = 0
    for id in sort(collect(keys(sizes)))
        off[id] = o; o += sizes[id]
    end
    return off, o
end

function affrow(Z, r::Int, off, m)
    g = zeros(m)
    for (bi, id) in enumerate(collect(Z.generator_ids))
        G = Z.Gs[bi]; o = off[id]
        @inbounds for c in 1:size(G, 2)
            g[o+c] += G[r, c]
        end
    end
    return AffRow(Z.c[r], g)
end

pick_dz(DZ, which::AbstractString) = which == "d" ? DZ.∂Z : which == "z1" ? DZ.Z₁ : which == "z2" ? DZ.Z₂ :
    kerr("precondition", "term variable must be d|z1|z2")

"""Deterministic list of linear claims `Σ coeff·v ≤ rhs` made by a lemma (interval parts, then the relational
part). Shared, by specification, with the checker (which re-implements it)."""
function lemma_specs(st::LemmaStatement)
    out = Tuple{Vector{Tuple{String,Int,Float64}},Float64}[]
    for (k, j) in enumerate(st.neurons)
        isfinite(st.dhi[k]) && push!(out, ([("d", j, 1.0)], st.dhi[k]))
        isfinite(st.dlo[k]) && push!(out, ([("d", j, -1.0)], -st.dlo[k]))
        if !isnothing(st.n1lo)
            isfinite(st.n1hi[k]) && push!(out, ([("z1", j, 1.0)], st.n1hi[k]))
            isfinite(st.n1lo[k]) && push!(out, ([("z1", j, -1.0)], -st.n1lo[k]))
        end
        if !isnothing(st.n2lo)
            isfinite(st.n2hi[k]) && push!(out, ([("z2", j, 1.0)], st.n2hi[k]))
            isfinite(st.n2lo[k]) && push!(out, ([("z2", j, -1.0)], -st.n2lo[k]))
        end
    end
    isempty(st.terms) || push!(out, (st.terms, st.rhs))
    return out
end

"""Affine row of Σ coeff·v over the zonotope DZ (a DiffZonotope)."""
function combo_row(DZ, terms, off, m)
    c = 0.0; g = zeros(m)
    for (w, j, a) in terms
        r = affrow(pick_dz(DZ, w), j, off, m)
        c += a * r.c; g .+= a .* r.g
    end
    return AffRow(c, g)
end

dual_bound(o::AffRow, A::Matrix{Float64}, r::Vector{Float64}, λ::Vector{Float64}) =
    isempty(λ) ? o.c + sum(abs, o.g) : o.c + sum(abs, o.g .- A' * λ) + dot(λ, r)

"""Propose λ ≥ 0 via GLPK; returns the λ with the smaller recomputed bound (λ = 0 if the LP fails).
`eqpairs` lists row pairs (i, k) with A[k,:] = −A[i,:] that encode an equality (with tiny outward slack in r).
GLPK copes badly with such almost-coincident half-spaces, so each pair is given to the solver as one equality
without offset; its free multiplier μ is mapped back onto the two certificate rows (λ_i, λ_k) = (μ⁺, μ⁻).
The returned λ is always re-evaluated with the certificate rows, so this only affects tightness."""
function propose_lambda(o::AffRow, A::Matrix{Float64}, r::Vector{Float64}; eqpairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[])
    nc = size(A, 1)
    nc == 0 && return Float64[]
    best = zeros(nc); bestb = dual_bound(o, A, r, best)
    try
        model = Model(GLPK.Optimizer); set_silent(model)
        set_time_limit_sec(model, LP_TIME_LIMIT[])        # GLPK can stall on degenerate LPs; no duals → λ = 0 (sound)
        set_attribute(model, "presolve", GLPK.GLP_ON)
        m = length(o.g)
        @variable(model, -1.0 <= e[1:m] <= 1.0)
        # Identity pairs hold only up to rounding and are numerically overdetermined: give the solver a band
        # |row − mid| ≤ τ (τ = 1e-7 relative) instead of an exact equality or two almost coincident half-spaces.
        rs = copy(r)
        for (i, k) in eqpairs
            τ = 1e-7 * (1.0 + abs(r[i]) + sum(abs, @view A[i, :]))
            mid = (r[i] - r[k]) / 2
            rs[i] = mid + τ; rs[k] = -mid + τ
        end
        cons = @constraint(model, A * e .<= rs)
        @objective(model, Max, sum(o.g[i] * e[i] for i in 1:m if o.g[i] != 0.0; init=0.0))
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL && has_duals(model)
            d = dual.(cons)
            for σ in (-1.0, 1.0)
                cand = max.(0.0, σ .* d)
                b = dual_bound(o, A, r, cand)
                b < bestb && (best = cand; bestb = b)
            end
        end
    catch e
        @warn "LP failed; falling back to λ = 0" exception = e
    end
    return best
end

"""Constraint rows (A, r) and the list of lemma ids used, from the in-scope lemmas whose layer zonotope is
available in this propagation."""
function lemma_constraints(S::Session, res::PropResult, lemma_ids::Vector{Int}, off, m;
                           constraints::Vector{LemmaStatement}=LemmaStatement[], used_cons::Vector{Int}=Int[],
                           relax::Bool=false, relax_spec::Vector{Any}=Any[], eqpairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[])
    rows = Vector{Float64}[]; r = Float64[]; used = Int[]
    for L in sort(lemma_ids)
        st = getlemma(S, L).stmt
        Zk = zono_at_relu_input(S, res, st.relu_layer)
        isnothing(Zk) && continue
        specs = lemma_specs(st)
        isempty(specs) && continue
        push!(used, L)
        for (terms, rhs) in specs
            row = combo_row(Zk, terms, off, m)
            push!(rows, row.g); push!(r, rhs - row.c)
        end
    end
    # case-split constraints of the node (after the lemma rows, in node order)
    for (i, st) in enumerate(constraints)
        Zk = zono_at_relu_input(S, res, st.relu_layer)
        isnothing(Zk) && continue
        specs = lemma_specs(st)
        isempty(specs) && continue
        push!(used_cons, i)
        for (terms, rhs) in specs
            row = combo_row(Zk, terms, off, m)
            push!(rows, row.g); push!(r, rhs - row.c)
        end
    end
    if relax
        rr, rv, spec = relax_rows(S, res, off, m)
        base = length(rows)
        append!(rows, rr); append!(r, rv); append!(relax_spec, spec)
        for (i, sp) in enumerate(spec)       # identity pairs: "+" row directly followed by its "−" row
            sp[3] in ("dpre+", "dpost+") && push!(eqpairs, (base + i, base + i + 1))
        end
    end
    A = isempty(rows) ? zeros(0, m) : permutedims(reduce(hcat, rows))
    return A, r, used
end

# ------------------------------------------------------------------ LP-relaxation rows
# True relations between concrete values at ReLU layer k, unit j (z = input, a = output; 1/2 = N1/N2; d, t = the
# differential input/output), all evaluated on the zonotope rows that one witness ε reproduces simultaneously:
#   a ≥ 0, a ≥ z, a ≤ chord of ReLU over [min(l,0), max(u,0)]        (cached bounds l ≤ z ≤ u)
#   d = z1 − z2, t = a1 − a2                                          (identities, two rows each)
#   min(0,d) ≤ t ≤ max(0,d), each side relaxed by its chord over [min(dl,0), max(du,0)]
# The formulas are valid for any sound bounds, so a checker with (slightly) different bounds rebuilds valid rows.
const RELAX_TYPES = ("a1>=0", "a1>=z1", "a1<=ch", "a2>=0", "a2>=z2", "a2<=ch", "dpre+", "dpre-", "dpost+", "dpost-", "t<=ch", "t>=ch")

chord_slope(l, u) = (l′ = min(l, 0.0); u′ = max(u, 0.0); u′ > l′ ? (u′ / (u′ - l′), l′) : (0.0, 0.0))

"""Row g·ε ≤ rhs for (layer k, unit j, type) from input/output zonotopes and the layer's cache bounds."""
function relax_row(Zi, Zo, bc, j::Int, typ::String, off, m)
    R(Z) = affrow(Z, j, off, m)
    lin(cs) = (g = zeros(m); c = 0.0; for (a, r) in cs; g .+= a .* r.g; c += a * r.c; end; (g, c))
    if typ in ("a1>=0", "a2>=0")
        a = typ == "a1>=0" ? R(Zo.Z₁) : R(Zo.Z₂); g, c = lin([(-1.0, a)]); k0 = 0.0
    elseif typ in ("a1>=z1", "a2>=z2")
        z, a = typ == "a1>=z1" ? (R(Zi.Z₁), R(Zo.Z₁)) : (R(Zi.Z₂), R(Zo.Z₂)); g, c = lin([(1.0, z), (-1.0, a)]); k0 = 0.0
    elseif typ in ("a1<=ch", "a2<=ch")
        z, a, l, u = typ == "a1<=ch" ? (R(Zi.Z₁), R(Zo.Z₁), bc.lower₁[j], bc.upper₁[j]) : (R(Zi.Z₂), R(Zo.Z₂), bc.lower₂[j], bc.upper₂[j])
        s, l′ = chord_slope(l, u); g, c = lin([(1.0, a), (-s, z)]); k0 = s * l′         # a − s z + s l′ ≤ 0
    elseif typ in ("dpre+", "dpre-")
        σ = typ == "dpre+" ? 1.0 : -1.0; g, c = lin([(σ, R(Zi.∂Z)), (-σ, R(Zi.Z₁)), (σ, R(Zi.Z₂))]); k0 = 0.0
    elseif typ in ("dpost+", "dpost-")
        σ = typ == "dpost+" ? 1.0 : -1.0; g, c = lin([(σ, R(Zo.∂Z)), (-σ, R(Zo.Z₁)), (σ, R(Zo.Z₂))]); k0 = 0.0
    elseif typ == "t<=ch"
        s, l′ = chord_slope(bc.∂lower[j], bc.∂upper[j]); g, c = lin([(1.0, R(Zo.∂Z)), (-s, R(Zi.∂Z))]); k0 = s * l′
    elseif typ == "t>=ch"
        s, l′ = chord_slope(bc.∂lower[j], bc.∂upper[j]); g, c = lin([(1.0 - s, R(Zi.∂Z)), (-1.0, R(Zo.∂Z))]); k0 = s * l′
    else
        kerr("internal", "unknown relaxation row type $typ")
    end
    rhs = -(c + k0)
    return g, rhs + 1e-12 * (1.0 + abs(rhs) + sum(abs, g))     # outward rounding allowance
end

"""Relaxation rows for all units that are unstable in N1, N2 or in d, at the ReLU layers this propagation ran."""
function relax_rows(S::Session, res::PropResult, off, m)
    rows = Vector{Float64}[]; r = Float64[]; spec = Any[]
    for k in S.relu_layers
        (k in res.layers_done) || continue
        !isnothing(res.stop) && k >= res.stop && continue
        L = layer_by_idx(S, k)
        Zi = res.ps.zono_storage.zonotopes[get_inputs(L)[1]].zonotope
        Zo = res.ps.zono_storage.zonotopes[get_outputs(L)[1]].zonotope
        bc = res.cache.bounds_cache[k]
        for j in eachindex(bc.lower₁)
            ((bc.lower₁[j] < 0 < bc.upper₁[j]) || (bc.lower₂[j] < 0 < bc.upper₂[j]) || (bc.∂lower[j] < 0 < bc.∂upper[j])) || continue
            for typ in RELAX_TYPES
                g, rhs = relax_row(Zi, Zo, bc, j, typ, off, m)
                push!(rows, g); push!(r, rhs); push!(spec, Any[k, j, typ])
            end
        end
    end
    return rows, r, spec
end

function relax_zonos(S::Session, res::PropResult)
    Zs = Any[]
    for k in S.relu_layers
        (k in res.layers_done) || continue
        !isnothing(res.stop) && k >= res.stop && continue
        L = layer_by_idx(S, k)
        for pos in (get_inputs(L)[1], get_outputs(L)[1])
            Z = res.ps.zono_storage.zonotopes[pos].zonotope; append!(Zs, [Z.∂Z, Z.Z₁, Z.Z₂])
        end
    end
    return Zs
end

"""Zonotope at the input of ReLU layer k if it was computed in this propagation, else nothing."""
function zono_at_relu_input(S::Session, res::PropResult, k::Int)
    stop = res.stop
    !isnothing(stop) && k > stop && return nothing
    !isnothing(stop) && k == stop && return res.Zout
    k in res.layers_done || return nothing
    pos = get_inputs(layer_by_idx(S, k))[1]
    z = res.ps.zono_storage.zonotopes[pos]
    return isnothing(z) ? nothing : z.zonotope
end

"""All zonotopes involved (for the variable layout)."""
function lp_zonos(S::Session, res::PropResult, lemma_ids, extra; constraints::Vector{LemmaStatement}=LemmaStatement[])
    Zs = Any[extra...]
    for k in vcat([getlemma(S, L).stmt.relu_layer for L in lemma_ids], [st.relu_layer for st in constraints])
        Zk = zono_at_relu_input(S, res, k)
        isnothing(Zk) || append!(Zs, [Zk.∂Z, Zk.Z₁, Zk.Z₂])
    end
    return Zs
end

"""
Try to establish `objective_k ≤ t_k` for all k with lemma constraints. `objs` is a vector of
(label, AffRow, threshold). Returns (ok, evidence) where evidence lists, for each objective that needed the LP,
its label and λ, plus the lemma ids whose constraints were used (in order).
"""
function lp_establish(S::Session, res::PropResult, lemma_ids::Vector{Int}, objs, off, m;
                      constraints::Vector{LemmaStatement}=LemmaStatement[], relax::Bool=false)
    used_cons = Int[]; relax_spec = Any[]; eqpairs = Tuple{Int,Int}[]
    A, r, used = lemma_constraints(S, res, lemma_ids, off, m; constraints=constraints, used_cons=used_cons,
                                   relax=relax, relax_spec=relax_spec, eqpairs=eqpairs)
    ev = Dict{String,Any}[]
    worst = -Inf
    for (label, o, t) in objs
        b0 = o.c + sum(abs, o.g)
        if b0 <= t
            continue
        end
        λ = propose_lambda(o, A, r; eqpairs=eqpairs)
        b = dual_bound(o, A, r, λ)
        worst = max(worst, b - t)
        push!(ev, Dict{String,Any}("o" => label, "lam" => λ))
    end
    evd = Dict{String,Any}("lemmas" => used, "cons" => used_cons, "rows" => ev)
    relax && (evd["relax"] = relax_spec)
    return worst <= 0.0, evd, worst
end
