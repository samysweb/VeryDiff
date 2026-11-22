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

"""Propose λ ≥ 0 via GLPK; returns the λ with the smaller recomputed bound (λ = 0 if the LP fails)."""
function propose_lambda(o::AffRow, A::Matrix{Float64}, r::Vector{Float64})
    nc = size(A, 1)
    nc == 0 && return Float64[]
    best = zeros(nc); bestb = dual_bound(o, A, r, best)
    try
        model = Model(GLPK.Optimizer); set_silent(model)
        m = length(o.g)
        @variable(model, -1.0 <= e[1:m] <= 1.0)
        cons = @constraint(model, A * e .<= r)
        @objective(model, Max, sum(o.g[i] * e[i] for i in 1:m if o.g[i] != 0.0; init=0.0))
        optimize!(model)
        if has_duals(model)
            d = dual.(cons)
            for cand in (max.(0.0, -d), max.(0.0, d))
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
                           constraints::Vector{LemmaStatement}=LemmaStatement[], used_cons::Vector{Int}=Int[])
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
    A = isempty(rows) ? zeros(0, m) : permutedims(reduce(hcat, rows))
    return A, r, used
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
                      constraints::Vector{LemmaStatement}=LemmaStatement[])
    used_cons = Int[]
    A, r, used = lemma_constraints(S, res, lemma_ids, off, m; constraints=constraints, used_cons=used_cons)
    ev = Dict{String,Any}[]
    worst = -Inf
    for (label, o, t) in objs
        b0 = o.c + sum(abs, o.g)
        if b0 <= t
            continue
        end
        λ = propose_lambda(o, A, r)
        b = dual_bound(o, A, r, λ)
        worst = max(worst, b - t)
        push!(ev, Dict{String,Any}("o" => label, "lam" => λ))
    end
    return worst <= 0.0, Dict{String,Any}("lemmas" => used, "cons" => used_cons, "rows" => ev), worst
end
