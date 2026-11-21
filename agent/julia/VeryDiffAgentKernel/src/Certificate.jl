# Versioned JSON certificate. Node encoding (compact):
#   [id, parent, side, kind, payload, region]
#     region  : [lo, hi] restricted to `active_dims` (redundant; the checker requires it to equal
#               the region reconstructed from the spec box and the cuts)
#     side    : 0 = obligation root or assume-child, 1 = lower child [lo,cut], 2 = upper child [cut,hi]
#     kind    : "s" split   payload {"b": block, "d": dim, "c": cut, "p": propagated?}
#               "a" assume  payload {"l": lemma, "p": propagated?}
#               "c" closed  payload {"db": distance_bound}
#               "o" open    payload {}
#   "p" = the node was propagated (checked) before being expanded, i.e. its children inherit
#   its post-propagation bounds cache; otherwise they inherit its inherited cache unchanged.
# The checker never trusts stored regions: it reconstructs them and compares.

function cert_nodes(S::Session)
    out = Any[]
    for id in sort(collect(keys(S.nodes)))
        n = S.nodes[id]
        p = !isnothing(n.cache_out)
        r = [n.region.lo[S.active_dims], n.region.hi[S.active_dims]]
        if n.kind == :split
            push!(out, [id, n.parent, n.side, "s", Dict("b" => string(n.split_block), "d" => n.split_dim, "c" => n.split_cut, "p" => p), r])
        elseif n.kind == :assume
            push!(out, [id, n.parent, n.side, "a", Dict("l" => n.lemma, "p" => p), r])
        elseif n.kind == :closed
            push!(out, [id, n.parent, n.side, "c", Dict("db" => encode_float(n.distance_bound)), r])
        else
            push!(out, [id, n.parent, n.side, "o", Dict(), r])
        end
    end
    return out
end

function cert_lemma(S::Session, lem::Lemma)
    s = lem.stmt
    d = Dict{String,Any}("id" => lem.id, "obligation" => lem.obligation, "scope_node" => lem.scope_node,
        "relu_layer" => s.relu_layer, "tensor" => S.pos_names[get_inputs(layer_by_idx(S, s.relu_layer))[1]],
        "neurons" => s.neurons, "dlo" => encode_floats(s.dlo), "dhi" => encode_floats(s.dhi),
        "status" => lemma_status(S, lem.id))
    isnothing(s.n1lo) || (d["n1"] = [encode_floats(s.n1lo), encode_floats(s.n1hi)])
    isnothing(s.n2lo) || (d["n2"] = [encode_floats(s.n2lo), encode_floats(s.n2hi)])
    isnothing(lem.refutation) || (d["refutation"] = lem.refutation)
    return d
end

function certificate(S::Session)
    validate_tree(S)
    Dict{String,Any}(
        "version" => CERT_VERSION,
        "kernel_version" => KERNEL_VERSION,
        "spec_hash" => S.spec_hash,
        "verydiff_commit" => S.verydiff_commit,
        "verydiff_src_tree" => S.verydiff_src_tree,
        "property" => JSON.parse(S.property_canon),
        "files" => Dict("net1" => basename(S.net1_path), "net2" => basename(S.net2_path), "spec" => basename(S.spec_path)),
        "softmax_removed" => S.softmax_removed,
        "status" => session_verdict(S),
        "spec_box" => Dict("lo" => S.root_box.lo, "hi" => S.root_box.hi),
        "active_dims" => S.active_dims,
        "obligations" => [Dict("id" => o.id, "kind" => string(o.kind), "root" => o.root,
                               "target" => o.kind == :main ? "property" : "lemma:$(o.lemma)")
                          for o in sort(collect(values(S.obligations)); by=o -> o.id)],
        "nodes" => cert_nodes(S),
        "lemmas" => [cert_lemma(S, l) for l in sort(collect(values(S.lemmas)); by=l -> l.id)],
        "cex" => isnothing(S.cex) ? nothing : Dict("x" => S.cex["x"], "n1" => S.cex["y1"], "n2" => S.cex["y2"], "dist" => S.cex["dist"]),
        "log" => S.log,
        "stats" => Dict("nodes" => length(S.nodes), "auto_nodes" => count(n -> n.created_by == :auto, values(S.nodes)),
                        "kernel_seconds" => round(S.kernel_time; digits=3), "propagations" => S.n_propagations))
end

function export_certificate(S::Session, path::AbstractString)
    t = time()
    c = certificate(S)
    s = JSON.json(c)
    if endswith(path, ".gz")
        open(path, "w") do io
            gz = GzipCompressorStream(io)
            write(gz, s)
            close(gz)
        end
    else
        write(path, s)
    end
    S.kernel_time += time() - t
    return Dict("path" => path, "status" => c["status"], "bytes" => filesize(path), "nodes" => length(c["nodes"]))
end
