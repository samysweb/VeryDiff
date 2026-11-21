# Proof actions. Every action validates its preconditions; none can assert closure.

function require_open_leaf(n::Node)
    n.kind == :open || kerr("precondition", "node $(n.id) is not an open leaf (kind=$(n.kind))")
end

obligation_of(S::Session, n::Node) = S.obligations[n.obligation]

"""Internal check. Returns (outcome::Symbol, PropResult, heur_info) with outcome ∈
:closed | :open | :cex | :refuted."""
function do_check!(S::Session, n::Node)
    require_open_leaf(n)
    obl = obligation_of(S, n)
    a = obl.adapter
    t = time()
    res = run_propagation(S, n.region, n.cache_in, n.lemmas_in_scope, stop_layer(a);
                          verification_status=n.verification_status)
    sat, cex, heur, status′, db = adapter_check(a, S.X1, S.X2, res.Zin, res.Zout, n.verification_status)
    n.cache_out = res.cache                         # owned: run_propagation deep-copied cache_in
    n.distance_bound = db
    guards_ok = isempty(res.inconsistent) && !res.nonfinite && !isnan(db)
    outcome = :open
    reason = ""
    if sat && guards_ok
        n.kind = :closed
        n.evidence = Dict{String,Any}("distance_bound" => db, "check" => check_tag(a))
        outcome = :closed
    else
        if sat && !guards_ok
            reason = !isempty(res.inconsistent) ? "inconsistent bounds cache (lower>upper) at ReLU layers $(res.inconsistent); refusing to close" :
                     "NaN in propagated bounds; refusing to close"
        else
            reason = "property not established by abstraction"
        end
        if obl.kind == :main && !isnothing(cex)
            x = clamp_to(n.region, Vector{Float64}(cex[1]))
            ok, r = genuine_cex(S, x)
            if !ok
                x = (n.region.lo .+ n.region.hi) ./ 2
                ok, r = genuine_cex(S, x)
            end
            if ok
                S.cex = Dict{String,Any}("x" => x, "y1" => r["y1"], "y2" => r["y2"], "dist" => r["dist"], "node" => n.id)
                n.cex = true
                outcome = :cex
                reason = "genuine counterexample (re-evaluated concretely)"
            else
                reason = "abstraction reported a cex that did not survive concrete re-evaluation"
            end
        elseif obl.kind == :lemma && !isnothing(heur)
            L = S.lemmas[obl.lemma]
            for x in heur[1]
                xc = clamp_to(n.region, Vector{Float64}(x))
                if refutes_lemma(S, L, xc)
                    L.refuted = true
                    L.refutation = Dict{String,Any}("x" => xc, "node" => n.id)
                    outcome = :refuted
                    reason = "lemma refuted by a concrete point; retract it"
                    break
                end
            end
        end
    end
    n.last_check = Dict{String,Any}("holds" => outcome == :closed, "distance_bound" => db,
        "outcome" => string(outcome), "reason" => reason, "seconds" => round(time() - t; digits=4),
        "inconsistent_layers" => res.inconsistent)
    return outcome, res, heur
end

function timed(f, S::Session)
    t = time()
    try
        return f()
    finally
        S.kernel_time += time() - t
    end
end

function check!(S::Session, node::Int; rationale::AbstractString="")
    timed(S) do
        n = getnode(S, node)
        log_action!(S, "check", Dict("node" => node), rationale)
        n.rationale = rationale
        outcome, _, _ = do_check!(S, n)
        r = copy(n.last_check)
        r["node"] = node
        r["verdict"] = session_verdict(S)
        return r
    end
end

"""Split an open leaf at `cut` (default: midpoint) along `dim`. `cuts` gives a k-way split,
recorded as a chain of binary splits."""
function do_split!(S::Session, n::Node, d::Int, cut::Union{Nothing,Float64}; dim_block::Symbol=:primary,
                   created_by::Symbol=:agent)
    require_open_leaf(n)
    a = obligation_of(S, n).adapter
    dim_block in split_blocks(a) || kerr("precondition", "split block $dim_block not supported by this property")
    dim_block == :primary || throw(NotImplementedError("split on block $dim_block (secondary/differential dimensions)"))
    1 <= d <= dim(n.region) || kerr("precondition", "dimension $d out of range 1:$(dim(n.region))")
    d in S.active_dims || kerr("precondition", "dimension $d is fixed (degenerate) in the spec")
    lo, hi = n.region.lo[d], n.region.hi[d]
    c = isnothing(cut) ? lo / 2 + hi / 2 : cut
    (lo < c < hi) || kerr("precondition", "cut $c must satisfy $lo < cut < $hi (region too small to split?)")
    b1, b2 = split_box(n.region, d, c)
    parent_cache = inheritable_cache(n)
    c1 = new_node!(S, n.obligation, n.id, 1, b1, n.lemmas_in_scope, deepcopy(parent_cache), n.work_share / 2,
                   n.depth + 1, created_by; distance_bound=n.distance_bound, verification_status=n.verification_status)
    c2 = new_node!(S, n.obligation, n.id, 2, b2, n.lemmas_in_scope, deepcopy(parent_cache), n.work_share / 2,
                   n.depth + 1, created_by; distance_bound=n.distance_bound, verification_status=n.verification_status)
    n.kind = :split
    n.split_block = dim_block
    n.split_dim = d
    n.split_cut = c
    n.children = [c1.id, c2.id]
    return c1, c2
end

function split!(S::Session, node::Int, d::Int; cut=nothing, cuts=nothing, dim_block="primary", rationale::AbstractString="")
    timed(S) do
        n = getnode(S, node)
        log_action!(S, "split", Dict("node" => node, "dim" => d, "cut" => cut, "cuts" => cuts, "dim_block" => dim_block), rationale)
        blk = Symbol(dim_block)
        if !isnothing(cuts)
            cs = Float64.(collect(cuts))
            (issorted(cs) && allunique(cs) && !isempty(cs)) || kerr("precondition", "cuts must be strictly increasing")
            all(n.region.lo[d] .< cs .< n.region.hi[d]) || kerr("precondition", "all cuts must lie strictly inside the region")
            leaves = Int[]
            cur = n
            for c in cs
                c1, c2 = do_split!(S, cur, d, c; dim_block=blk)
                c1.rationale = rationale
                push!(leaves, c1.id)
                cur = c2
            end
            push!(leaves, cur.id)
            n.rationale = rationale
            return Dict("node" => node, "children" => leaves)
        end
        c1, c2 = do_split!(S, n, d, isnothing(cut) ? nothing : Float64(cut); dim_block=blk)
        n.rationale = rationale
        return Dict("node" => node, "children" => [c1.id, c2.id], "cut" => n.split_cut)
    end
end

"""Delete the subtree below `n` (children, and lemmas introduced inside it together with their trees)."""
function discard_below!(S::Session, n::Node)
    for c in copy(n.children)
        for id in subtree(S, c)
            m = S.nodes[id]
            if m.kind == :assume
                delete_lemma!(S, m.lemma)
            end
            delete!(S.nodes, id)
        end
    end
    if n.kind == :assume
        delete_lemma!(S, n.lemma)
        n.lemma = 0
    end
    empty!(n.children)
end

function delete_lemma!(S::Session, L::Int)
    haskey(S.lemmas, L) || return
    lem = S.lemmas[L]
    obl = S.obligations[lem.obligation]
    root = S.nodes[obl.root]
    discard_below!(S, root)
    delete!(S.nodes, root.id)
    delete!(S.obligations, obl.id)
    delete!(S.lemmas, L)
end

"""Backtracking: collapse the subtree under `node` back to an open leaf. Only discards evidence."""
function prune!(S::Session, node::Int; rationale::AbstractString="")
    timed(S) do
        n = getnode(S, node)
        log_action!(S, "prune", Dict("node" => node), rationale)
        n.kind == :open && return Dict("node" => node, "removed" => 0)
        before = length(S.nodes)
        discard_below!(S, n)
        n.kind = :open
        n.evidence = nothing
        n.split_dim = 0; n.split_cut = NaN; n.split_block = :primary
        n.rationale = rationale
        return Dict("node" => node, "removed" => before - length(S.nodes), "verdict" => session_verdict(S))
    end
end

function submit_counterexample!(S::Session, x::Vector{Float64}; rationale::AbstractString="")
    timed(S) do
        log_action!(S, "submit_counterexample", Dict("x" => x), rationale)
        contains_point(S.root_box, x) || return Dict("accepted" => false, "reason" => "point outside the spec box")
        ok, r = genuine_cex(S, x)
        if !ok
            return Dict("accepted" => false, "reason" => "no violation at this point", "dist" => r["dist"])
        end
        S.cex = Dict{String,Any}("x" => x, "y1" => r["y1"], "y2" => r["y2"], "dist" => r["dist"], "node" => 0)
        return Dict("accepted" => true, "dist" => r["dist"], "verdict" => session_verdict(S))
    end
end
