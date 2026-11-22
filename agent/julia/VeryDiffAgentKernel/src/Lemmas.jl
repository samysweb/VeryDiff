# ∂-bounds lemmas at a ReLU input.

parse_bounds(v, n) = isnothing(v) ? nothing : (length(v) == n ? Float64[decode_float(x) for x in v] :
    kerr("precondition", "bounds vector has length $(length(v)), expected $n"))

function introduce_lemma!(S::Session, node::Int, relu_layer::Int, neurons::Vector{Int}, dlo, dhi;
                          n1_bounds=nothing, n2_bounds=nothing, rationale::AbstractString="")
    timed(S) do
        n = getnode(S, node)
        log_action!(S, "introduce_lemma", Dict("node" => node, "relu_layer" => relu_layer, "neurons" => neurons,
            "dlo" => encode_floats(dlo), "dhi" => encode_floats(dhi), "n1_bounds" => n1_bounds, "n2_bounds" => n2_bounds), rationale)
        require_open_leaf(n)
        relu_layer in S.relu_layers || kerr("precondition", "layer $relu_layer is not a ReLU layer (ReLU layers: $(S.relu_layers))")
        width = length(relu_width(S, relu_layer))
        k = length(neurons)
        (k > 0 && allunique(neurons) && all(1 .<= neurons .<= width)) || kerr("precondition", "neurons must be distinct indices in 1:$width")
        lo = parse_bounds(dlo, k); hi = parse_bounds(dhi, k)
        (isnothing(lo) || isnothing(hi)) && kerr("precondition", "dlo and dhi are required")
        all(lo .<= hi) || kerr("precondition", "dlo must be <= dhi")
        n1lo = n1hi = n2lo = n2hi = nothing
        if !isnothing(n1_bounds)
            n1lo = parse_bounds(n1_bounds[1], k); n1hi = parse_bounds(n1_bounds[2], k)
            all(n1lo .<= n1hi) || kerr("precondition", "n1 lower must be <= upper")
        end
        if !isnothing(n2_bounds)
            n2lo = parse_bounds(n2_bounds[1], k); n2hi = parse_bounds(n2_bounds[2], k)
            all(n2lo .<= n2hi) || kerr("precondition", "n2 lower must be <= upper")
        end
        stmt = LemmaStatement(relu_layer, copy(neurons), lo, hi, n1lo, n1hi, n2lo, n2hi)
        Lid = S.next_lemma; S.next_lemma += 1
        # Lemma obligation (lemma chaining): same region; it inherits the lemmas already in scope at `n` and
        # n's bounds cache (which depends only on those lemmas). Non-circular: those lemmas were introduced
        # earlier and their proofs cannot refer to L. L counts as proven only if they are proven as well.
        root = new_node!(S, 0, 0, 0, n.region, copy(n.lemmas_in_scope), deepcopy(inheritable_cache(n)), 1.0, 0, :agent)
        obl = new_obligation!(S, :lemma, root.id, Lid, LemmaBoundsAdapter(stmt))
        root.obligation = obl
        S.lemmas[Lid] = Lemma(Lid, stmt, n.id, obl, false, nothing, rationale)
        # main side: node becomes :assume(L) with one child in which L is in scope
        child = new_node!(S, n.obligation, n.id, 0, n.region, vcat(n.lemmas_in_scope, [Lid]),
                          deepcopy(inheritable_cache(n)), n.work_share, n.depth + 1, :agent;
                          distance_bound=n.distance_bound, verification_status=n.verification_status)
        n.kind = :assume
        n.lemma = Lid
        n.children = [child.id]
        n.rationale = rationale
        return Dict("lemma" => Lid, "assume_node" => n.id, "child" => child.id, "lemma_root" => root.id)
    end
end

relu_width(S::Session, k::Int) = begin
    # width of the ReLU input = output dim of the preceding layer; obtain it from a cheap propagation-free
    # source: the cache shape if known, else the weight matrix of the nearest preceding Linear layer.
    layers = get_layers(S.N)
    i = findfirst(l -> l.layer_idx == k, layers)
    for j in i-1:-1:1
        l = layers[j].layer1
        if hasproperty(l, :dense)
            return 1:size(l.dense.weight, 1)
        end
    end
    return 1:dim(S.root_box)
end

"""Phase lemma: units `neurons` of ReLU layer `relu_layer` are inactive ("off": pre-activation ≤ 0) or active
("on": ≥ 0) on region(node), in net1 and/or net2. Encoded as one-sided N1/N2 bounds with unconstrained ∂."""
function introduce_phase_lemma!(S::Session, node::Int, relu_layer::Int, neurons::Vector{Int};
                                net1::AbstractString="any", net2::AbstractString="any", rationale::AbstractString="")
    k = length(neurons)
    side(p) = p == "off" ? (fill(-Inf, k), zeros(k)) : p == "on" ? (zeros(k), fill(Inf, k)) :
              p == "any" ? nothing : kerr("precondition", "phase must be off|on|any")
    b1, b2 = side(net1), side(net2)
    (isnothing(b1) && isnothing(b2)) && kerr("precondition", "give a phase for net1 and/or net2")
    r = Dict{String,Any}(introduce_lemma!(S, node, relu_layer, neurons, fill(-Inf, k), fill(Inf, k);
                         n1_bounds=isnothing(b1) ? nothing : [b1[1], b1[2]],
                         n2_bounds=isnothing(b2) ? nothing : [b2[1], b2[2]],
                         rationale=isempty(rationale) ? "phase lemma net1=$net1 net2=$net2" : rationale))
    r["phase"] = Dict("net1" => net1, "net2" => net2)
    return r
end

function prove_lemma!(S::Session, L::Int; max_nodes::Int=1000, max_seconds::Float64=60.0, rationale::AbstractString="")
    lem = getlemma(S, L)
    lem.refuted && kerr("precondition", "lemma $L is refuted; retract it")
    log_action!(S, "prove_lemma", Dict("lemma" => L, "max_nodes" => max_nodes, "max_seconds" => max_seconds), rationale)
    r = auto_expand!(S, S.obligations[lem.obligation].root; max_nodes=max_nodes, max_seconds=max_seconds, log=false)
    r["lemma"] = L
    r["lemma_status"] = haskey(S.lemmas, L) ? lemma_status(S, L) : "deleted"
    return r
end

function retract_lemma!(S::Session, L::Int; rationale::AbstractString="")
    timed(S) do
        lem = getlemma(S, L)
        log_action!(S, "retract_lemma", Dict("lemma" => L), rationale)
        n = S.nodes[lem.scope_node]
        n.kind == :assume && n.lemma == L || kerr("internal", "lemma $L scope node is not its assume node")
        before = length(S.nodes)
        discard_below!(S, n)          # deletes child subtree, the lemma and its proof tree
        n.kind = :open
        n.rationale = rationale
        return Dict("lemma" => L, "reopened_node" => n.id, "removed_nodes" => before - length(S.nodes))
    end
end

"""Kernel-checked refutation by a concrete point (e.g. found in the sandbox)."""
function refute_lemma!(S::Session, L::Int, x::Vector{Float64}; rationale::AbstractString="")
    timed(S) do
        lem = getlemma(S, L)
        log_action!(S, "refute_lemma", Dict("lemma" => L, "x" => x), rationale)
        if refutes_lemma(S, lem, x)
            lem.refuted = true
            lem.refutation = Dict{String,Any}("x" => x, "node" => 0)
            return Dict("refuted" => true)
        end
        return Dict("refuted" => false, "reason" => "point outside the lemma region or bounds hold there")
    end
end

# JSON cannot carry ±Inf; accept strings "inf"/"-inf"/"Infinity"/"-Infinity".
function decode_float(x)
    x isa Number && return Float64(x)
    s = lowercase(string(x))
    s in ("inf", "+inf", "infinity", "+infinity") && return Inf
    s in ("-inf", "-infinity") && return -Inf
    return parse(Float64, s)
end
encode_float(x::Real) = isfinite(x) ? Float64(x) : (x > 0 ? "Infinity" : (x < 0 ? "-Infinity" : "NaN"))
encode_floats(v) = isnothing(v) ? nothing : [x isa Real ? encode_float(x) : x for x in v]

# ------------------------------------------------------------- dry runs
snapshot(S::Session) = (deepcopy(S.nodes), deepcopy(S.obligations), deepcopy(S.lemmas), S.next_node, S.next_obl,
                        S.next_lemma, deepcopy(S.cex), length(S.log), S.n_propagations, S.n_checks)
function restore!(S::Session, s)
    S.nodes, S.obligations, S.lemmas, S.next_node, S.next_obl, S.next_lemma, S.cex = s[1:7]
    resize!(S.log, s[8])
    used = S.n_propagations - s[9]
    S.n_propagations, S.n_checks = s[9], s[10]
    S.n_trial_propagations += used
    return used
end

"""
Dry run of a lemma at `node` (an open leaf): introduce it, prove it with `lemma_max_nodes`, then run auto_expand
on the assume child with `main_max_nodes`; optionally (`baseline`) also auto_expand `node` without the lemma with
the same budget. Everything is undone afterwards; the propagations are counted as trial work, not proof cost.
`kind` = "interval" (dlo/dhi, n1_bounds, n2_bounds) or "phase" (net1/net2 = off|on|any).
"""
function trial_lemma!(S::Session, node::Int, relu_layer::Int, neurons::Vector{Int}; kind::AbstractString="phase",
                      dlo=nothing, dhi=nothing, n1_bounds=nothing, n2_bounds=nothing, net1="any", net2="any",
                      lemma_max_nodes::Int=2000, main_max_nodes::Int=2000, max_seconds::Float64=120.0,
                      baseline::Bool=true, rationale::AbstractString="")
    timed(S) do
        require_open_leaf(getnode(S, node))
        t0 = time()
        out = Dict{String,Any}("node" => node, "relu_layer" => relu_layer, "n_neurons" => length(neurons), "kind" => kind)
        snap = snapshot(S)
        try
            r = kind == "phase" ? introduce_phase_lemma!(S, node, relu_layer, neurons; net1=net1, net2=net2) :
                kind == "interval" ? introduce_lemma!(S, node, relu_layer, neurons,
                    isnothing(dlo) ? fill(-Inf, length(neurons)) : dlo, isnothing(dhi) ? fill(Inf, length(neurons)) : dhi;
                    n1_bounds=n1_bounds, n2_bounds=n2_bounds) :
                kerr("precondition", "kind must be phase|interval")
            c0 = S.n_checks
            p = prove_lemma!(S, r["lemma"]; max_nodes=lemma_max_nodes, max_seconds=max_seconds)
            out["lemma"] = Dict("status" => p["lemma_status"], "proof_nodes" => length(subtree(S, r["lemma_root"])),
                                "checks" => S.n_checks - c0, "open_leaves" => p["open_leaves"])
            c1 = S.n_checks
            a = auto_expand!(S, r["child"]; max_nodes=main_max_nodes, max_seconds=max_seconds, log=false)
            out["with_lemma"] = Dict("subtree_nodes" => length(subtree(S, r["child"])), "checks" => S.n_checks - c1,
                                     "closed" => a["subtree_closed"], "open_leaves" => a["open_leaves"])
        finally
            out["trial_propagations"] = restore!(S, snap)
        end
        if baseline
            snap = snapshot(S)
            try
                c1 = S.n_checks
                a = auto_expand!(S, node; max_nodes=main_max_nodes, max_seconds=max_seconds, log=false)
                out["without_lemma"] = Dict("subtree_nodes" => length(subtree(S, node)), "checks" => S.n_checks - c1,
                                            "closed" => a["subtree_closed"], "open_leaves" => a["open_leaves"])
            finally
                out["trial_propagations"] += restore!(S, snap)
            end
        end
        if haskey(out, "with_lemma") && get(out, "without_lemma", nothing) !== nothing &&
           out["lemma"]["status"] == "proven" && out["with_lemma"]["closed"] && out["without_lemma"]["closed"]
            out["net_saving_nodes"] = out["without_lemma"]["subtree_nodes"] - out["with_lemma"]["subtree_nodes"] - out["lemma"]["proof_nodes"]
        else
            out["net_saving_nodes"] = nothing
            out["note"] = "budgets too small to compare (something did not close); numbers are lower bounds"
        end
        out["seconds"] = round(time() - t0; digits=2)
        log_action!(S, "trial_lemma", Dict("node" => node, "relu_layer" => relu_layer, "neurons" => neurons, "kind" => kind,
            "result" => Dict(k => out[k] for k in ("lemma", "with_lemma", "without_lemma", "net_saving_nodes") if haskey(out, k))), rationale)
        return out
    end
end
