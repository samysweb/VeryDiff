# Versioned JSON certificate. Node encoding (compact):
#   [id, parent, side, kind, payload, region]
#     region  : [lo, hi] restricted to `active_dims` (redundant; the checker requires it to equal
#               the region reconstructed from the spec box and the cuts)
#     side    : 0 = obligation root or assume-child, 1 = lower child [lo,cut], 2 = upper child [cut,hi]
#     kind    : "s" split   payload {"b": block, "d": dim, "c": cut, "p": propagated?}
#               "a" assume  payload {"l": lemma, "p": propagated?}
#               "s" relu-split payload {"b": "relu", "k": ReLU layer, "v": "z1"|"z2"|"d", "j": neuron, "c": t, "p": ...}
#                   child side 1 assumes v <= t, side 2 assumes v >= t (same region; constraints are inherited)
#               "c" closed  payload {"db": distance_bound} (+ "lp": dual evidence, or "inf": true for a node whose
#                   bounds at some ReLU input are empty under its constraints)
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
        if n.kind == :split && n.split_block == :relu
            push!(out, [id, n.parent, n.side, "s", Dict("b" => "relu", "k" => n.bsplit[1], "v" => n.bsplit[2], "j" => n.bsplit[3],
                                                        "c" => n.split_cut, "p" => p), r])
        elseif n.kind == :split && n.split_block == :lin
            push!(out, [id, n.parent, n.side, "s", Dict("b" => "lin", "k" => n.lsplit[1], "terms" => [[v, j, a] for (v, j, a) in n.lsplit[2]],
                                                        "c" => n.split_cut, "p" => p), r])
        elseif n.kind == :split
            push!(out, [id, n.parent, n.side, "s", Dict("b" => string(n.split_block), "d" => n.split_dim, "c" => n.split_cut, "p" => p), r])
        elseif n.kind == :assume
            push!(out, [id, n.parent, n.side, "a", Dict("l" => n.lemma, "p" => p), r])
        elseif n.kind == :closed
            pl = Dict{String,Any}("db" => encode_float(n.distance_bound))
            !isnothing(n.evidence) && haskey(n.evidence, "lp") && (pl["lp"] = n.evidence["lp"])
            !isnothing(n.evidence) && get(n.evidence, "check", "") == "infeasible" && (pl["inf"] = true)
            push!(out, [id, n.parent, n.side, "c", pl, r])
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
    isempty(s.terms) || (d["terms"] = [[w, j, a] for (w, j, a) in s.terms]; d["rhs"] = s.rhs)
    isnothing(lem.refutation) || (d["refutation"] = lem.refutation)
    return d
end

function certificate(S::Session)
    validate_tree(S)
    Dict{String,Any}(
        "version" => CERT_VERSION,
        "kernel_version" => KERNEL_VERSION,
        "property" => JSON.parse(S.property_canon),
        "files" => Dict("net1" => basename(S.net1_path), "net2" => basename(S.net2_path), "spec" => basename(S.spec_path)),
        "softmax_removed" => S.softmax_removed,
        "options" => Dict("csb" => S.csb, "fallback" => S.fallback, "lp_relax" => S.lp_relax,
                         "bound_rows" => S.bound_rows, "bound_rows_rtol" => S.bound_rows_rtol,
                         "fallback_rule" => S.fallback_rule),
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
                        "kernel_seconds" => round(S.kernel_time; digits=3), "propagations" => S.n_propagations,
                        "checks" => S.n_checks, "inspection_propagations" => S.n_propagations - S.n_checks,
                        "trial_propagations" => S.n_trial_propagations))
end

function export_certificate(S::Session, path::AbstractString)
    t = time()
    c = certificate(S)
    s = JSON.json(sanitize(c))      # ±Inf (e.g. phase lemmas) are encoded as "Infinity"/"-Infinity"
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
