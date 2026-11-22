# Session: immutable spec + mutable proof state.

mutable struct Session
    net1_path::String
    net2_path::String
    spec_path::String
    N::Any                            # GeminiNetwork (final Softmax popped)
    E1::Any                           # OnnxNet for concrete evaluation
    E2::Any
    X1::Any                           # VeryDiff `Network` call objects (used by properties)
    X2::Any
    softmax_removed::Bool
    out_tensor::String                # tensor whose value is "the output" (pre-softmax if removed)
    pos_names::Dict{Int,String}       # zonotope storage position -> ONNX tensor name
    relu_layers::Vector{Int}
    root_box::Box
    active_dims::Vector{Int}
    property::PropertyAdapter
    property_canon::String
    spec_hash::String
    verydiff_commit::String
    verydiff_src_tree::String
    nodes::Dict{Int,Node}
    obligations::Dict{Int,Obligation}
    lemmas::Dict{Int,Lemma}
    next_node::Int
    next_obl::Int
    next_lemma::Int
    main_obligation::Int
    cex::Union{Nothing,Dict{String,Any}}
    log::Vector{Dict{String,Any}}
    kernel_time::Float64
    t0::Float64
    n_propagations::Int
end

const SESSION = Ref{Union{Nothing,Session}}(nothing)
current() = isnothing(SESSION[]) ? kerr("no_session", "no open session; call open_session first") : SESSION[]

# ------------------------------------------------------------- hashing / provenance
"""Canonical JSON: sorted keys, no whitespace, shortest round-trip floats."""
function canonical_json(x)
    if x isa AbstractDict
        ks = sort!(collect(string.(keys(x))))
        return "{" * join([JSON.json(k) * ":" * canonical_json(x[k]) for k in ks], ",") * "}"
    elseif x isa AbstractVector
        return "[" * join(canonical_json.(x), ",") * "]"
    elseif x isa AbstractFloat
        isfinite(x) || kerr("property", "non-finite number in property")
        return repr(Float64(x))
    else
        return JSON.json(x)
    end
end

function verydiff_root()
    return normpath(joinpath(pkgdir(VeryDiff)))
end

"""(HEAD commit, tree hash of src/ with a -dirty suffix if src/ has local changes)."""
function verydiff_provenance()
    root = verydiff_root()
    commit = try strip(read(`git -C $root rev-parse HEAD`, String)) catch; "unknown" end
    tree = try strip(read(`git -C $root rev-parse HEAD:src`, String)) catch; "unknown" end
    dirty = try !isempty(strip(read(`git -C $root status --porcelain -- src`, String))) catch; true end
    return String(commit), String(tree) * (dirty ? "-dirty" : "")
end

"""sha256 over (net1 bytes, net2 bytes, spec bytes, canonical property JSON, VeryDiff src tree).
Each component is length-prefixed so the encoding is unambiguous."""
function compute_spec_hash(net1::String, net2::String, spec::String, property_canon::String, src_tree::String)
    ctx = SHA.SHA256_CTX()
    for part in (read(net1), read(net2), read(spec), Vector{UInt8}(property_canon), Vector{UInt8}(src_tree))
        SHA.update!(ctx, Vector{UInt8}(string(length(part), ":")))
        SHA.update!(ctx, part)
    end
    return bytes2hex(SHA.digest!(ctx))
end

# ------------------------------------------------------------- spec loading
"""Load (net1, net2, spec) like Cli.jl/verify_network; returns pieces needed by the session.
Shared with nothing in the checker (the checker re-does this independently)."""
function load_problem(net1::String, net2::String, spec::String)
    isfile(net1) || kerr("io", "file not found: $net1")
    isfile(net2) || kerr("io", "file not found: $net2")
    isfile(spec) || kerr("io", "file not found: $spec")
    o1 = load_onnx_model(net1)
    o2 = load_onnx_model(net2)
    ast, n_inputs, _ = get_ast(spec)
    boxes = Matrix{Float64}[]
    for (bounds, _, _, _) in ast
        push!(boxes, Matrix{Float64}(bounds[1:n_inputs, :]))
    end
    length(boxes) == 1 || kerr("spec", "MVP supports exactly one input box per spec (found $(length(boxes)))")
    b = boxes[1]
    root = Box(Vector{Float64}(b[:, 1]), Vector{Float64}(b[:, 2]))
    N = GeminiNetwork(o1, o2)
    softmax_removed = false
    last = N.diff_layers[end]
    out_pos = get_outputs(last)[1]
    if last.layer1 isa ONNXSoftmax
        pop!(N.diff_layers)
        softmax_removed = true
        out_pos = get_outputs(N.diff_layers[end])[1]
    end
    length(N.inputs) == 1 || kerr("spec", "only single-input networks are supported")
    _, io_map = sort_network(o1)
    pos_names = Dict{Int,String}(v => string(k) for (k, v) in io_map)
    relu = [l.layer_idx for l in N.diff_layers if l.layer1 isa ONNXRelu]
    return (o1=o1, o2=o2, N=N, root=root, softmax_removed=softmax_removed,
            out_tensor=pos_names[out_pos], pos_names=pos_names, relu=relu)
end

function open_session(net1::String, net2::String, spec::String, property::AbstractDict)
    t = time()
    VeryDiff.USE_DIFFZONO[] = true
    VeryDiff.NEW_HEURISTIC[] = true
    adapter = make_property(property)
    P = load_problem(net1, net2, spec)
    canon = canonical_json(property_json(adapter))
    commit, tree = verydiff_provenance()
    h = compute_spec_hash(net1, net2, spec, canon, tree)
    active = findall(P.root.lo .< P.root.hi)
    S = Session(abspath(net1), abspath(net2), abspath(spec), P.N, P.o1, P.o2, executable_network(P.o1), executable_network(P.o2), P.softmax_removed,
        P.out_tensor, P.pos_names, P.relu, P.root, active, adapter, canon, h, commit, tree,
        Dict{Int,Node}(), Dict{Int,Obligation}(), Dict{Int,Lemma}(), 1, 1, 1, 0, nothing,
        Dict{String,Any}[], 0.0, time(), 0)
    root = new_node!(S, 0, 0, 0, P.root, Int[], TaskBounds(), 1.0, 0, :agent)
    S.main_obligation = new_obligation!(S, :main, root.id, 0, adapter)
    root.obligation = S.main_obligation
    SESSION[] = S
    S.kernel_time += time() - t
    return S
end

function new_node!(S::Session, obligation, parent, side, region, lemmas, cache_in, work_share, depth, created_by;
                   distance_bound=Inf, verification_status=nothing)
    n = Node(S.next_node, obligation, parent, side, region, lemmas, cache_in, work_share, depth, created_by;
             distance_bound=distance_bound, verification_status=verification_status)
    S.nodes[n.id] = n
    S.next_node += 1
    return n
end

function new_obligation!(S::Session, kind, root, lemma, adapter)
    o = Obligation(S.next_obl, kind, root, lemma, adapter)
    S.obligations[o.id] = o
    S.next_obl += 1
    return o.id
end

getnode(S::Session, id::Integer) = haskey(S.nodes, id) ? S.nodes[id] : kerr("bad_node", "no node with id $id")
getlemma(S::Session, id::Integer) = haskey(S.lemmas, id) ? S.lemmas[id] : kerr("bad_lemma", "no lemma with id $id")

# ------------------------------------------------------------- derived closure
"""closed(node) ⟺ closed leaf, or all children closed and (for :assume) the lemma is proven.
There is no way to assert closure."""
function is_closed(S::Session, id::Int; stack::Set{Int}=Set{Int}())
    n = S.nodes[id]
    if n.kind == :closed
        return true
    elseif n.kind == :open
        return false
    elseif n.kind == :split
        return all(c -> is_closed(S, c; stack=stack), n.children)
    elseif n.kind == :assume
        length(n.children) == 1 || kerr("internal", "assume node $(n.id) must have exactly one child")
        return is_closed(S, n.children[1]; stack=stack) && lemma_proven(S, n.lemma; stack=stack)
    end
    kerr("internal", "unknown node kind $(n.kind)")
end

function lemma_proven(S::Session, L::Int; stack::Set{Int}=Set{Int}())
    haskey(S.lemmas, L) || return false
    lem = S.lemmas[L]
    lem.refuted && return false
    L in stack && kerr("internal", "circular lemma dependency through lemma $L")
    push!(stack, L)
    root = S.obligations[lem.obligation].root
    r = is_closed(S, root; stack=stack) && all(L2 -> lemma_proven(S, L2; stack=stack), S.nodes[root].lemmas_in_scope)
    delete!(stack, L)
    return r
end

lemma_status(S::Session, L::Int) = S.lemmas[L].refuted ? "refuted" : (lemma_proven(S, L) ? "proven" : "open")

function session_verdict(S::Session)
    !isnothing(S.cex) && return "UNSAFE"
    is_closed(S, S.obligations[S.main_obligation].root) && return "SAFE"
    return "PARTIAL"
end

"""All node ids in the subtree rooted at `id` (pre-order)."""
function subtree(S::Session, id::Int)
    out = Int[]
    st = [id]
    while !isempty(st)
        x = pop!(st)
        push!(out, x)
        append!(st, reverse(S.nodes[x].children))
    end
    return out
end

"""Structural invariants (used by tests and before export)."""
function validate_tree(S::Session)
    for (id, n) in S.nodes
        o = S.obligations[n.obligation]
        if n.parent == 0
            o.root == id || kerr("internal", "orphan node $id")
        else
            p = S.nodes[n.parent]
            p.obligation == n.obligation || kerr("internal", "node $id crosses obligations")
            id in p.children || kerr("internal", "node $id not among its parent's children")
        end
        if n.kind == :split
            length(n.children) == 2 || kerr("internal", "split node $id needs 2 children")
            c1, c2 = S.nodes[n.children[1]], S.nodes[n.children[2]]
            (a, b) = split_box(n.region, n.split_dim, n.split_cut)
            (c1.region == a && c2.region == b) || kerr("internal", "split node $id children do not partition it")
        elseif n.kind == :assume
            length(n.children) == 1 || kerr("internal", "assume node $id needs 1 child")
            S.nodes[n.children[1]].region == n.region || kerr("internal", "assume child region differs")
            n.lemma in S.nodes[n.children[1]].lemmas_in_scope || kerr("internal", "assume child misses lemma")
        else
            isempty(n.children) || kerr("internal", "leaf $id has children")
        end
        # non-circularity: every in-scope lemma was introduced at an ancestor in the same obligation, or is
        # inherited by the obligation root (lemma chaining: exactly the lemmas in scope at its assume node).
        root_id = S.obligations[n.obligation].root
        inherited = S.nodes[root_id].lemmas_in_scope
        if n.id == root_id && o.kind == :lemma
            sn = S.nodes[S.lemmas[o.lemma].scope_node]
            Set(inherited) == Set(sn.lemmas_in_scope) || kerr("internal", "lemma root $id inherits a wrong lemma scope")
            o.lemma in inherited && kerr("internal", "lemma $(o.lemma) in scope of its own proof")
        end
        for L in n.lemmas_in_scope
            L in inherited && continue
            lem = S.lemmas[L]
            a = S.nodes[lem.scope_node]
            a.obligation == n.obligation || kerr("internal", "lemma $L used outside the obligation it was introduced in")
            x = n.parent; found = false
            while x != 0
                x == a.id && (found = true; break)
                x = S.nodes[x].parent
            end
            found || kerr("internal", "lemma $L in scope at node $id but not introduced on its path")
        end
    end
    return true
end

function log_action!(S::Session, action::String, args::AbstractDict, rationale::AbstractString)
    push!(S.log, Dict{String,Any}("t" => round(time() - S.t0; digits=3), "action" => action,
        "args" => args, "rationale" => String(rationale)))
end
