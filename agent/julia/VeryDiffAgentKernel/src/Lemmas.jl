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
        # lemma obligation: same region, fresh cache, no lemmas in scope (non-circular by construction)
        root = new_node!(S, 0, 0, 0, n.region, Int[], TaskBounds(), 1.0, 0, :agent)
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
