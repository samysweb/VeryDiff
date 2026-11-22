# Read-only inspection. Nothing here mutates the proof tree (propagations use deep copies).

r6(x::Real) = isfinite(x) ? round(Float64(x); sigdigits=6) : encode_float(x)
r6(v::AbstractVector) = [r6(x) for x in v]

"""Token-frugal summary of interval vectors."""
function summarize_bounds(lo::AbstractVector, hi::AbstractVector; neurons=nothing, limit::Int=16, topk::Int=5)
    n = length(lo)
    if !isnothing(neurons)
        all(1 .<= neurons .<= n) || kerr("precondition", "neuron index out of range 1:$n")
        return Dict("n" => n, "neurons" => neurons, "lo" => r6(lo[neurons]), "hi" => r6(hi[neurons]))
    end
    if n <= limit
        return Dict("n" => n, "lo" => r6(lo), "hi" => r6(hi))
    end
    w = hi .- lo
    idx = sortperm(w; rev=true)[1:min(topk, n)]
    return Dict("n" => n, "min_lo" => r6(minimum(lo)), "max_hi" => r6(maximum(hi)),
        "mean_width" => r6(mean(w)), "max_abs" => r6(maximum(max.(abs.(lo), abs.(hi)))),
        "widest" => [Dict("i" => i, "lo" => r6(lo[i]), "hi" => r6(hi[i])) for i in idx])
end

function closed_map(S::Session)
    memo = Dict{Int,Bool}()
    lmemo = Dict{Int,Bool}()
    function f(id)
        haskey(memo, id) && return memo[id]
        n = S.nodes[id]
        v = n.kind == :closed ? true : n.kind == :open ? false :
            n.kind == :split ? all(f, n.children) :
            (f(n.children[1]) && g(n.lemma))
        memo[id] = v
        return v
    end
    function g(L)
        haskey(lmemo, L) && return lmemo[L]
        lem = S.lemmas[L]
        lmemo[L] = false       # cycle guard (cannot happen by construction)
        v = !lem.refuted && f(S.obligations[lem.obligation].root)
        lmemo[L] = v
        return v
    end
    for id in keys(S.nodes); f(id); end
    for L in keys(S.lemmas); g(L); end
    return memo, lmemo
end

"""Closed volume within an obligation: sum of work_share over maximal closed subtrees.
`conditional` ignores lemma status (counts leaves closed under assumed lemmas)."""
function closed_volume(S::Session, root::Int, memo; conditional=false)
    v = 0.0
    st = [root]
    while !isempty(st)
        n = S.nodes[pop!(st)]
        if memo[n.id]
            v += n.work_share
        elseif n.kind == :closed
            v += n.work_share
        elseif conditional || n.kind == :split
            append!(st, n.children)
        end
    end
    return v
end

function session_status(S::Session)
    memo, lmemo = closed_map(S)
    mroot = S.obligations[S.main_obligation].root
    main_nodes = subtree(S, mroot)
    leaves = [S.nodes[i] for i in main_nodes if is_leaf(S.nodes[i])]
    Dict{String,Any}(
        "verdict" => session_verdict(S),
        "spec_hash" => S.spec_hash,
        "property" => JSON.parse(S.property_canon),
        "main_root" => mroot,
        "main" => Dict("nodes" => length(main_nodes), "open_leaves" => count(n -> n.kind == :open, leaves),
            "closed_leaves" => count(n -> n.kind == :closed, leaves),
            "closed_volume_fraction" => r6(closed_volume(S, mroot, memo)),
            "conditionally_closed_volume_fraction" => r6(closed_volume(S, mroot, memo; conditional=true)),
            "auto_nodes" => count(i -> S.nodes[i].created_by == :auto, main_nodes)),
        "lemmas" => [Dict("id" => L, "status" => (lem.refuted ? "refuted" : lmemo[L] ? "proven" : "open"),
                          "relu_layer" => lem.stmt.relu_layer, "n_neurons" => length(lem.stmt.neurons),
                          "assume_node" => lem.scope_node, "proof_root" => S.obligations[lem.obligation].root,
                          "proof_open_leaves" => count(i -> S.nodes[i].kind == :open, subtree(S, S.obligations[lem.obligation].root)))
                     for (L, lem) in sort(collect(S.lemmas); by=first)],
        "cex" => isnothing(S.cex) ? nothing : Dict("x" => r6(S.cex["x"]), "dist" => r6(S.cex["dist"])),
        "total_nodes" => length(S.nodes),
        "kernel_seconds" => round(S.kernel_time; digits=2),
        "propagations" => S.n_propagations)
end

function layer_table(S::Session)
    w = dim(S.root_box)
    rows = Dict{String,Any}[]
    for l in get_layers(S.N)
        L1 = l.layer1
        typ = string(nameof(typeof(L1)))
        if hasproperty(L1, :dense)
            w = size(L1.dense.weight, 1)
        end
        push!(rows, Dict("idx" => l.layer_idx, "type" => typ, "width" => w, "inputs" => l.inputs, "outputs" => l.outputs,
            "is_relu" => L1 isa ONNXRelu, "input_tensor" => S.pos_names[l.inputs[1]],
            "diff_zero" => string(nameof(typeof(l.diff_layer))) == "ZeroDense"))
    end
    return rows
end
list_layers(S::Session) = Dict("layers" => layer_table(S), "relu_layers" => S.relu_layers,
    "input_dim" => dim(S.root_box), "softmax_removed" => S.softmax_removed)

function node_brief(S::Session, n::Node)
    Dict("id" => n.id, "depth" => n.depth, "distance_bound" => r6(n.distance_bound),
         "volume" => r6(n.work_share), "created_by" => string(n.created_by))
end

function list_open_leaves(S::Session; sort_by::AbstractString="distance_bound", limit::Int=10, offset::Int=0,
                          obligation::Union{Nothing,Int}=nothing, under::Union{Nothing,Int}=nothing)
    root = !isnothing(under) ? under :
           S.obligations[isnothing(obligation) ? S.main_obligation : obligation].root
    leaves = [S.nodes[i] for i in subtree(S, root) if S.nodes[i].kind == :open]
    key = sort_by == "distance_bound" ? (n -> -n.distance_bound) :
          sort_by == "volume" ? (n -> -n.work_share) :
          sort_by == "depth" ? (n -> n.depth) : kerr("precondition", "sort_by must be distance_bound|volume|depth")
    sort!(leaves; by=n -> (key(n), n.id))
    sel = leaves[min(offset + 1, length(leaves) + 1):min(offset + limit, length(leaves))]
    return Dict("total" => length(leaves), "offset" => offset, "leaves" => [node_brief(S, n) for n in sel])
end

function region_summary(S::Session, b::Box; limit::Int=16)
    dims = length(S.active_dims) <= limit ? S.active_dims :
           [i for i in S.active_dims if b.lo[i] != S.root_box.lo[i] || b.hi[i] != S.root_box.hi[i]]
    return Dict("dims" => dims, "lo" => b.lo[dims], "hi" => b.hi[dims],
                "note" => length(dims) < length(S.active_dims) ? "only dims differing from the spec box shown" : "")
end

function get_node(S::Session, id::Int)
    n = getnode(S, id)
    path = Int[]
    x = id
    while x != 0
        pushfirst!(path, x); x = S.nodes[x].parent
    end
    obl = S.obligations[n.obligation]
    d = Dict{String,Any}("id" => id, "kind" => string(n.kind), "obligation" => n.obligation,
        "obligation_kind" => string(obl.kind), "closed" => is_closed(S, id), "region" => region_summary(S, n.region),
        "path" => length(path) > 20 ? vcat(path[1:3], [-1], path[end-15:end]) : path,
        "children" => n.children, "lemmas_in_scope" => n.lemmas_in_scope, "last_check" => n.last_check,
        "rationale" => n.rationale, "created_by" => string(n.created_by), "volume" => r6(n.work_share),
        "distance_bound" => r6(n.distance_bound), "has_cache_out" => !isnothing(n.cache_out))
    n.kind == :split && (d["split"] = Dict("block" => string(n.split_block), "dim" => n.split_dim, "cut" => n.split_cut))
    n.kind == :assume && (d["lemma"] = n.lemma)
    n.kind == :closed && (d["evidence"] = n.evidence)
    return d
end

"""Re-propagate node (from its inherited cache, with its in-scope lemma seeds, exactly as `check` would)."""
inspect_propagation(S::Session, n::Node; stop=nothing) =
    run_propagation(S, n.region, n.cache_in, n.lemmas_in_scope, stop; verification_status=n.verification_status)

function pick_zono(Z, which::AbstractString)
    which == "d" ? Z.∂Z : which == "z1" ? Z.Z₁ : which == "z2" ? Z.Z₂ : kerr("precondition", "which must be d|z1|z2")
end

function generators_topk(Zc, neurons, topk)
    out = Dict{String,Any}[]
    for r in neurons
        entries = Tuple{Float64,Int,Int}[]
        for (bi, G) in enumerate(Zc.Gs)
            for c in 1:size(G, 2)
                G[r, c] != 0.0 && push!(entries, (G[r, c], Zc.generator_ids[bi], c))
            end
        end
        sort!(entries; by=e -> -abs(e[1]))
        push!(out, Dict("neuron" => r, "center" => r6(Zc.c[r]), "n_generators" => length(entries),
            "top" => [Dict("g" => r6(e[1]), "block" => e[2], "col" => e[3]) for e in entries[1:min(topk, end)]]))
    end
    return out
end

function inspect_layer(S::Session, id::Int, layer_idx::Int; which::AbstractString="d", neurons=nothing,
                       detail::AbstractString="bounds", topk::Int=5, limit::Int=16)
    n = getnode(S, id)
    L = layer_by_idx(S, layer_idx)
    res = inspect_propagation(S, n)
    layer_idx in res.layers_done || kerr("precondition", "layer $layer_idx was not propagated")
    Zo = res.ps.zono_storage.zonotopes[get_outputs(L)[1]].zonotope
    Zi = res.ps.zono_storage.zonotopes[get_inputs(L)[1]].zonotope
    zo = pick_zono(Zo, which)
    b = zono_bounds(zo)
    d = Dict{String,Any}("node" => id, "layer" => layer_idx, "type" => string(nameof(typeof(L.layer1))), "which" => which,
        "output_bounds" => summarize_bounds(b[:, 1], b[:, 2]; neurons=neurons, limit=limit, topk=topk))
    if L.layer1 isa ONNXRelu
        bi = zono_bounds(pick_zono(Zi, which))
        d["input_bounds"] = summarize_bounds(bi[:, 1], bi[:, 2]; neurons=neurons, limit=limit, topk=topk)
        sel = get_selectors(zono_bounds(Zi.Z₁), zono_bounds(Zi.Z₂), zono_bounds(Zi.∂Z))
        names = ("zero_diff", "neg_neg", "neg_pos", "pos_neg", "pos_pos", "any_neg", "neg_any", "any_pos", "pos_any", "any_any")
        d["selector_counts"] = Dict(nm => count(s) for (nm, s) in zip(names, sel))
        b1 = zono_bounds(Zi.Z₁); b2 = zono_bounds(Zi.Z₂)
        d["unstable"] = Dict("n1" => count((b1[:, 1] .< 0) .& (b1[:, 2] .> 0)),
                             "n2" => count((b2[:, 1] .< 0) .& (b2[:, 2] .> 0)),
                             "d_relaxed" => count(sel[8] .| sel[9] .| sel[10]))
        if detail == "generators_topk"
            isnothing(neurons) && kerr("precondition", "detail=generators_topk needs explicit neurons (<= 8)")
            length(neurons) <= 8 || kerr("precondition", "at most 8 neurons for generators_topk")
            d["input_generators"] = generators_topk(pick_zono(Zi, which), neurons, topk)
        end
    end
    if detail == "generators_topk" && !(L.layer1 isa ONNXRelu)
        isnothing(neurons) && kerr("precondition", "detail=generators_topk needs explicit neurons (<= 8)")
        length(neurons) <= 8 || kerr("precondition", "at most 8 neurons for generators_topk")
        d["output_generators"] = generators_topk(zo, neurons, topk)
    end
    return d
end

function output_bounds(S::Session, id::Int; neurons=nothing, limit::Int=16)
    n = getnode(S, id)
    res = inspect_propagation(S, n)
    bd = zono_bounds(res.Zout.∂Z); b1 = zono_bounds(res.Zout.Z₁); b2 = zono_bounds(res.Zout.Z₂)
    return Dict("node" => id, "d" => summarize_bounds(bd[:, 1], bd[:, 2]; neurons=neurons, limit=limit),
        "z1" => summarize_bounds(b1[:, 1], b1[:, 2]; neurons=neurons, limit=limit),
        "z2" => summarize_bounds(b2[:, 1], b2[:, 2]; neurons=neurons, limit=limit),
        "distance_bound" => r6(maximum(abs.(bd))))
end

function split_scores(S::Session, id::Int; top_k::Int=5)
    n = getnode(S, id)
    a = obligation_of(S, n).adapter
    res = inspect_propagation(S, n; stop=stop_layer(a))
    w = split_weights(a, res.Zin, res.Zout)
    dims = res.task.distance_indices
    order = sortperm(w; rev=true)[1:min(top_k, length(w))]
    return Dict("node" => id, "heuristic" => "verydiff (influence-weighted generator mass)",
        "scores" => [Dict("dim" => dims[i], "score" => r6(w[i]), "lo" => n.region.lo[dims[i]], "hi" => n.region.hi[dims[i]]) for i in order])
end

function cache_bounds(S::Session, id::Int, relu_layer::Int; neurons=nothing, limit::Int=16)
    n = getnode(S, id)
    relu_layer in S.relu_layers || kerr("precondition", "layer $relu_layer is not a ReLU layer")
    function summ(tb)
        isnothing(tb) && return nothing
        haskey(tb.bounds_cache, relu_layer) || return "absent"
        bc = tb.bounds_cache[relu_layer]
        bc.initialized || return "uninitialized"
        return Dict("d" => summarize_bounds(bc.∂lower, bc.∂upper; neurons=neurons, limit=limit),
                    "n1" => summarize_bounds(bc.lower₁, bc.upper₁; neurons=neurons, limit=limit),
                    "n2" => summarize_bounds(bc.lower₂, bc.upper₂; neurons=neurons, limit=limit))
    end
    return Dict("node" => id, "relu_layer" => relu_layer,
        "cache_in" => summ(n.cache_in), "cache_out" => summ(n.cache_out),
        "note" => "cache_in is what the next check of this node starts from; children inherit cache_out if present")
end

# ------------------------------------------------------------- exports (full data, for the sandbox)
function export_data(S::Session, what::AbstractString, node_ids, path::AbstractString)
    ids = isnothing(node_ids) ? Int[] : Int.(collect(node_ids))
    data = Dict{String,Any}("what" => what, "spec_hash" => S.spec_hash)
    if what == "tree"
        data["nodes"] = [Dict("id" => n.id, "parent" => n.parent, "obligation" => n.obligation, "kind" => string(n.kind),
            "lo" => n.region.lo, "hi" => n.region.hi, "distance_bound" => encode_float(n.distance_bound),
            "created_by" => string(n.created_by), "split_dim" => n.split_dim, "lemma" => n.lemma)
            for n in sort(collect(values(S.nodes)); by=n -> n.id)]
        data["obligations"] = [Dict("id" => o.id, "kind" => string(o.kind), "root" => o.root, "lemma" => o.lemma) for o in values(S.obligations)]
    elseif what == "regions"
        data["regions"] = [Dict("id" => i, "lo" => getnode(S, i).region.lo, "hi" => getnode(S, i).region.hi) for i in ids]
        data["spec_box"] = Dict("lo" => S.root_box.lo, "hi" => S.root_box.hi)
    elseif what == "bounds"
        data["nodes"] = Dict{String,Any}()
        for i in ids
            res = inspect_propagation(S, getnode(S, i))
            layers = Dict{String,Any}()
            for l in get_layers(S.N)
                Z = res.ps.zono_storage.zonotopes[get_outputs(l)[1]].zonotope
                layers[string(l.layer_idx)] = Dict(k => (b = zono_bounds(pick_zono(Z, k)); Dict("lo" => b[:, 1], "hi" => b[:, 2]))
                                                   for k in ("d", "z1", "z2"))
            end
            data["nodes"][string(i)] = layers
        end
    elseif what == "generators"
        data["nodes"] = Dict{String,Any}()
        for i in ids
            res = inspect_propagation(S, getnode(S, i))
            Z = res.Zout
            data["nodes"][string(i)] = Dict(k => Dict("c" => pick_zono(Z, k).c, "ids" => collect(pick_zono(Z, k).generator_ids),
                "G" => [[collect(row) for row in eachrow(G)] for G in pick_zono(Z, k).Gs]) for k in ("d", "z1", "z2"))
            data["nodes"][string(i)]["input"] = Dict("c" => res.Zin.Z₁.c, "G" => [collect(r) for r in eachrow(res.Zin.Z₁.Gs[1])],
                "active_dims" => res.task.distance_indices)
        end
    else
        kerr("precondition", "what must be tree|regions|bounds|generators")
    end
    open(path, "w") do io
        JSON.json(io, data)
    end
    return Dict("path" => path, "bytes" => filesize(path))
end

# ------------------------------------------------------------- weights (read-only, "weights as text")
"""Weights of one affine layer for net1, net2 or their difference (net1 − net2), plus a structural
summary of the difference (where it lives: nonzero rows/cols, norms, leading singular values)."""
function read_weights(S::Session, layer_idx::Int; which::AbstractString="diff", rows=nothing, cols=nothing,
                      max_entries::Int=400, digits::Int=4)
    L = layer_by_idx(S, layer_idx)
    l1, l2 = L.layer1, L.layer2
    if hasproperty(l1, :dense)
        W1, b1, W2, b2 = l1.dense.weight, copy(l1.dense.bias), l2.dense.weight, copy(l2.dense.bias)
        # ONNX MatMul+Add graphs put the bias into a separate AddConst layer: fold it in so the affine map is complete.
        layers = get_layers(S.N)
        i = findfirst(x -> x.layer_idx == layer_idx, layers)
        bias_note = "bias of this layer"
        if i < length(layers) && hasproperty(layers[i+1].layer1, :c) && !hasproperty(layers[i+1].layer1, :dense)
            b1 .+= layers[i+1].layer1.c; b2 .+= layers[i+1].layer2.c
            bias_note = "bias folded in from the following AddConst layer $(layers[i+1].layer_idx)"
        end
    elseif hasproperty(l1, :c)
        W1 = W2 = nothing; b1, b2 = l1.c, l2.c
    else
        kerr("precondition", "layer $layer_idx ($(nameof(typeof(l1)))) has no weights")
    end
    pick(a, b) = which == "net1" ? a : which == "net2" ? b : which == "diff" ? (isnothing(a) ? nothing : a .- b) :
                 kerr("precondition", "which must be net1|net2|diff")
    W = pick(W1, W2); b = pick(b1, b2)
    d = Dict{String,Any}("layer" => layer_idx, "type" => string(nameof(typeof(l1))), "which" => which)
    if !isnothing(W1)
        ΔW = W1 .- W2; Δb = b1 .- b2   # b1/b2 include a folded AddConst bias (see b_note)
        sv = svdvals(Matrix(ΔW))
        d["shape"] = collect(size(W1))
        d["diff_summary"] = Dict("frobenius" => r6(norm(ΔW)), "rel_frobenius" => r6(norm(ΔW) / max(norm(W1), eps())),
            "nonzero_entries" => count(!iszero, ΔW), "nonzero_rows" => findall(r -> any(!iszero, r), collect(eachrow(ΔW)))[1:min(end, 64)],
            "n_nonzero_rows" => count(r -> any(!iszero, r), eachrow(ΔW)), "n_nonzero_cols" => count(c -> any(!iszero, c), eachcol(ΔW)),
            "top_singular_values" => r6(sv[1:min(5, end)]), "bias_diff_max_abs" => r6(maximum(abs.(Δb))),
            "row_l1_max" => r6(maximum(sum(abs.(ΔW); dims=2))))
        R = isnothing(rows) ? (1:size(W, 1)) : rows
        C = isnothing(cols) ? (1:size(W, 2)) : cols
        all(r -> 1 <= r <= size(W, 1), R) && all(c -> 1 <= c <= size(W, 2), C) || kerr("precondition", "rows/cols out of range")
        if length(R) * length(C) <= max_entries
            d["rows"] = collect(R); d["cols"] = collect(C)
            d["W"] = [round.(W[r, collect(C)]; digits=digits) for r in R]
        else
            d["W"] = "omitted ($(length(R))×$(length(C)) > max_entries=$max_entries); pass rows/cols or use the sandbox"
        end
        d["b"] = round.(b[collect(R)]; digits=digits)
        d["b_note"] = bias_note
    else
        d["c"] = round.(b; digits=digits)
    end
    return d
end
