#!/usr/bin/env julia
# Independent replay checker for VeryDiff-Agent certificates.
#
#   julia --project=agent/julia/VeryDiffAgentKernel agent/checker/check_certificate.jl CERT NET1 NET2 SPEC
#
# Imports VeryDiff (transformers, epsilon property, VNNLib parsing) but NOT the kernel's
# tree / region / closure / seeding code: those bits are re-implemented here.
# Prints "ACCEPT <status> ..." or "REJECT <reason>" and exits 0 / 1.

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

struct Reject <: Exception
    msg::String
end
reject(msg) = throw(Reject(msg))
need(c, msg) = c || reject(msg)

# ---------------------------------------------------------------- numbers / hashing
dec(x) = x isa Number ? Float64(x) :
    (s = lowercase(string(x)); s == "infinity" ? Inf : s == "-infinity" ? -Inf : reject("bad number $x"))

function canon(x)
    if x isa AbstractDict
        ks = sort!(collect(string.(keys(x))))
        return "{" * join([JSON.json(k) * ":" * canon(x[k]) for k in ks], ",") * "}"
    elseif x isa AbstractVector
        return "[" * join(canon.(x), ",") * "]"
    elseif x isa AbstractFloat
        return repr(Float64(x))
    elseif x isa Integer
        return repr(Float64(x))      # the kernel stores property numbers as Float64
    else
        return JSON.json(x)
    end
end

function src_tree()
    root = pkgdir(VeryDiff)
    tree = try strip(read(`git -C $root rev-parse HEAD:src`, String)) catch; "unknown" end
    dirty = try !isempty(strip(read(`git -C $root status --porcelain -- src`, String))) catch; true end
    return String(tree) * (dirty ? "-dirty" : "")
end

function spec_hash(net1, net2, spec, prop_canon, tree)
    ctx = SHA.SHA256_CTX()
    for part in (read(net1), read(net2), read(spec), Vector{UInt8}(prop_canon), Vector{UInt8}(tree))
        SHA.update!(ctx, Vector{UInt8}(string(length(part), ":")))
        SHA.update!(ctx, part)
    end
    return bytes2hex(SHA.digest!(ctx))
end

# ---------------------------------------------------------------- boxes (re-implemented)
"""Outward centre/radius: widen by one ulp, round radius up, assert exact coverage."""
function task_for(lo::Vector{Float64}, hi::Vector{Float64}, active::Vector{Int}, tb::TaskBounds)
    mid = copy(lo); dist = Float64[]
    for i in active
        l = prevfloat(lo[i]); h = nextfloat(hi[i])
        m = l / 2 + h / 2
        d = max(nextfloat(m - l), nextfloat(h - m))
        R = Rational{BigInt}
        need(R(m) - R(d) <= R(lo[i]) && R(m) + R(d) >= R(hi[i]), "internal: coverage failure")
        mid[i] = m; push!(dist, d)
    end
    return VerificationTask(mid, dist, copy(active), nothing, nothing, nothing, nothing, nothing, Inf, 1.0, tb)
end

# ---------------------------------------------------------------- propagation with seeds (re-implemented)
struct Seed
    neurons::Vector{Int}; dlo::Vector{Float64}; dhi::Vector{Float64}
    n1::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}
    n2::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}
end

function seed!(bc::BoundsCache, n::Int, seeds::Vector{Seed})
    if !bc.initialized
        bc.lower₁ = fill(-Inf, n); bc.upper₁ = fill(Inf, n); bc.lower₂ = fill(-Inf, n); bc.upper₂ = fill(Inf, n)
        bc.∂lower = fill(-Inf, n); bc.∂upper = fill(Inf, n); bc.initialized = true
    end
    for s in seeds
        need(all(1 .<= s.neurons .<= n), "lemma neuron out of range")
        bc.∂lower[s.neurons] .= max.(bc.∂lower[s.neurons], s.dlo)
        bc.∂upper[s.neurons] .= min.(bc.∂upper[s.neurons], s.dhi)
        if !isnothing(s.n1)
            bc.lower₁[s.neurons] .= max.(bc.lower₁[s.neurons], s.n1[1]); bc.upper₁[s.neurons] .= min.(bc.upper₁[s.neurons], s.n1[2])
        end
        if !isnothing(s.n2)
            bc.lower₂[s.neurons] .= max.(bc.lower₂[s.neurons], s.n2[1]); bc.upper₂[s.neurons] .= min.(bc.upper₂[s.neurons], s.n2[2])
        end
    end
end

"""Propagate; returns (Zin, Zout_or_stop_input, cache, ok_guards)."""
function propagate(N, lo, hi, active, cache_in::TaskBounds, seeds::Dict{Int,Vector{Seed}}, stop)
    tb = deepcopy(cache_in)
    task = task_for(lo, hi, active, tb)
    P = PropState(true)
    prepare_prop_state!(P, task)
    layers = get_layers(N)
    max_io = maximum(max(maximum(l.inputs), maximum(l.outputs)) for l in layers)
    max_io > length(P.zono_storage.zonotopes) && resize_zonotope_storage!(P.zono_storage, max_io)
    Zout = nothing
    for L in layers
        if !isnothing(stop) && L.layer_idx == stop
            Zout = P.zono_storage.zonotopes[get_inputs(L)[1]].zonotope
            break
        end
        ins = get_zonos_at_pos(get_inputs(L), P)
        first_pass(P) && configure_first_usage!(L.layer_idx, ins)
        zonos_initialized(P, get_outputs(L)) || init_layer!(P, L, ins, get_outputs(L))
        izs = get_zonotope.(ins)
        outs = get_zonos_at_pos(get_outputs(L), P)
        bc = get!(() -> BoundsCache(), tb.bounds_cache, L.layer_idx)
        if haskey(seeds, L.layer_idx)
            need(L.layer1 isa ONNXRelu, "lemma on non-ReLU layer")
            seed!(bc, length(izs[1].Z₁.c), seeds[L.layer_idx])
        end
        propagate_layer!(outs, L, izs; bounds_cache=bc)
    end
    isnothing(stop) && (Zout = P.zono_storage.zonotopes[get_outputs(layers[end])[1]].zonotope)
    isnothing(Zout) && reject("stop layer $stop not found")
    ok = true
    for bc in values(tb.bounds_cache)
        bc.initialized || continue
        v = vcat(bc.lower₁, bc.upper₁, bc.lower₂, bc.upper₂, bc.∂lower, bc.∂upper)
        any(isnan, v) && (ok = false)
        (any(bc.lower₁ .> bc.upper₁) || any(bc.lower₂ .> bc.upper₂) || any(bc.∂lower .> bc.∂upper)) && (ok = false)
    end
    for Z in (Zout.∂Z, Zout.Z₁, Zout.Z₂)
        any(isnan, zono_bounds(Z)) && (ok = false)
    end
    return P.zono_storage.zonotopes[1].zonotope, Zout, tb, ok
end

# ---------------------------------------------------------------- the check
function load_cert(path)
    s = endswith(path, ".gz") ? String(transcode(GzipDecompressor, read(path))) : read(path, String)
    return JSON.parse(s)
end

function check(cert_path, net1, net2, specf; verbose=false)
    VeryDiff.USE_DIFFZONO[] = true
    VeryDiff.NEW_HEURISTIC[] = true
    C = load_cert(cert_path)
    need(get(C, "version", 0) == 1, "unsupported certificate version")
    prop = C["property"]
    need(get(prop, "kind", "") == "epsilon", "checker supports only epsilon properties")
    eps = Float64(prop["epsilon"])
    need(isfinite(eps) && eps >= 0, "bad epsilon")
    pc = canon(Dict("kind" => "epsilon", "epsilon" => eps))
    tree = src_tree()
    need(C["verydiff_src_tree"] == tree, "VeryDiff src tree mismatch (cert $(C["verydiff_src_tree"]), local $tree)")
    need(spec_hash(net1, net2, specf, pc, tree) == C["spec_hash"], "spec hash mismatch")

    # problem
    o1 = load_onnx_model(net1); o2 = load_onnx_model(net2)
    ast, n_inputs, _ = get_ast(specf)
    boxes = Matrix{Float64}[]
    for (b, _, _, _) in ast        # (iterator has no length; no comprehension)
        push!(boxes, Matrix{Float64}(b[1:n_inputs, :]))
    end
    need(length(boxes) == 1, "spec must have exactly one input box")
    lo0 = boxes[1][:, 1]; hi0 = boxes[1][:, 2]
    need(Float64.(C["spec_box"]["lo"]) == lo0 && Float64.(C["spec_box"]["hi"]) == hi0, "spec box mismatch")
    active = findall(lo0 .< hi0)
    need(Int.(C["active_dims"]) == active, "active dims mismatch")
    N = GeminiNetwork(o1, o2)
    _, io_map = sort_network(o1)
    names = Dict(v => string(k) for (k, v) in io_map)
    sm = N.diff_layers[end].layer1 isa ONNXSoftmax
    sm && pop!(N.diff_layers)
    need(sm == C["softmax_removed"], "softmax_removed mismatch")
    out_tensor = names[get_outputs(N.diff_layers[end])[1]]
    relu = Set(l.layer_idx for l in N.diff_layers if l.layer1 isa ONNXRelu)
    epsprop = VeryDiff.get_epsilon_property(eps)
    X1 = executable_network(o1); X2 = executable_network(o2)

    status = C["status"]
    need(status in ("SAFE", "UNSAFE", "PARTIAL"), "bad status")

    # UNSAFE: concrete re-evaluation only
    if status == "UNSAFE"
        cx = C["cex"]
        need(!isnothing(cx), "UNSAFE without cex")
        x = Float64.(cx["x"])
        need(length(x) == length(lo0) && all(lo0 .<= x .<= hi0), "cex outside the spec box")
        y1 = Float64.(vec(compute_all_outputs(o1, x)[out_tensor])); y2 = Float64.(vec(compute_all_outputs(o2, x)[out_tensor]))
        d = maximum(abs.(y1 .- y2))
        need(d > eps, "cex does not violate the property (dist $d <= eps $eps)")
        return "ACCEPT UNSAFE dist=$(d) eps=$(eps)"
    end
    status == "SAFE" && need(isnothing(C["cex"]), "SAFE certificate carries a cex")

    # structure
    nodes = Dict{Int,Any}()
    for e in C["nodes"]
        id = Int(e[1])
        need(!haskey(nodes, id), "duplicate node id $id")
        nodes[id] = (id=id, parent=Int(e[2]), side=Int(e[3]), kind=String(e[4]), p=e[5], r=e[6])
    end
    children = Dict{Int,Vector{Int}}(id => Int[] for id in keys(nodes))
    for (id, n) in nodes
        n.parent == 0 && continue
        need(haskey(nodes, n.parent), "node $id has missing parent $(n.parent)")
        push!(children[n.parent], id)
    end
    obls = Dict{Int,Any}()
    for o in C["obligations"]
        obls[Int(o["id"])] = (id=Int(o["id"]), kind=String(o["kind"]), root=Int(o["root"]), target=String(o["target"]))
    end
    mains = [o for o in values(obls) if o.kind == "main"]
    need(length(mains) == 1, "need exactly one main obligation")
    lemmas = Dict{Int,Any}()
    for l in C["lemmas"]
        id = Int(l["id"])
        k = length(l["neurons"])
        n1 = haskey(l, "n1") ? (dec.(l["n1"][1]), dec.(l["n1"][2])) : nothing
        n2 = haskey(l, "n2") ? (dec.(l["n2"][1]), dec.(l["n2"][2])) : nothing
        s = Seed(Int.(l["neurons"]), dec.(l["dlo"]), dec.(l["dhi"]), n1, n2)
        need(length(s.dlo) == k && length(s.dhi) == k, "lemma $id malformed")
        need(Int(l["relu_layer"]) in relu, "lemma $id on a non-ReLU layer")
        lemmas[id] = (id=id, obligation=Int(l["obligation"]), scope=Int(l["scope_node"]), layer=Int(l["relu_layer"]), seed=s)
        need(haskey(obls, lemmas[id].obligation) && obls[lemmas[id].obligation].kind == "lemma" &&
             obls[lemmas[id].obligation].target == "lemma:$id", "lemma $id has no matching obligation")
    end
    for o in values(obls)
        need(haskey(nodes, o.root) && nodes[o.root].parent == 0 && nodes[o.root].side == 0, "obligation $(o.id) root invalid")
        o.kind == "lemma" && need(any(l.obligation == o.id for l in values(lemmas)), "lemma obligation $(o.id) without lemma")
    end
    roots = Set(o.root for o in values(obls))
    for (id, n) in nodes
        n.parent == 0 && need(id in roots, "orphan node $id")
        ch = children[id]
        if n.kind == "s"
            need(length(ch) == 2 && Set(nodes[c].side for c in ch) == Set([1, 2]), "split node $id needs a lower and an upper child")
            need(n.p["b"] == "primary", "split block $(n.p["b"]) not supported")
        elseif n.kind == "a"
            need(length(ch) == 1 && nodes[ch[1]].side == 0, "assume node $id needs exactly one child")
            L = Int(n.p["l"])
            need(haskey(lemmas, L) && lemmas[L].scope == id, "assume node $id references an unknown lemma $L")
        elseif n.kind in ("c", "o")
            need(isempty(ch), "leaf $id has children")
        else
            reject("unknown node kind $(n.kind)")
        end
    end
    for l in values(lemmas)
        need(haskey(nodes, l.scope) && nodes[l.scope].kind == "a" && Int(nodes[l.scope].p["l"]) == l.id, "lemma $(l.id) scope node invalid")
    end

    # replay, top-down, per obligation
    region = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    obl_of = Dict{Int,Int}()
    verified_leaf = Dict{Int,Bool}()
    region[mains[1].root] = (copy(lo0), copy(hi0))
    order = Int[]                       # obligations in an order where lemma roots get their region first
    pending = collect(values(obls))
    # a lemma obligation's root region equals its scope node's region; process main first, then
    # lemma obligations whose scope node has been reached.
    function replay(o)
        stack = [(o.root, TaskBounds(), Int[])]
        adapter_stop = o.kind == "main" ? nothing : lemmas[parse(Int, split(o.target, ":")[2])].layer
        lem_self = o.kind == "main" ? nothing : lemmas[parse(Int, split(o.target, ":")[2])]
        while !isempty(stack)
            id, cache, scope = pop!(stack)
            haskey(obl_of, id) && reject("node $id reached twice")
            obl_of[id] = o.id
            n = nodes[id]
            lo, hi = region[id]
            need(Float64.(n.r[1]) == lo[active] && Float64.(n.r[2]) == hi[active], "stored region of node $id differs from reconstruction")
            seeds = Dict{Int,Vector{Seed}}()
            for L in scope
                push!(get!(seeds, lemmas[L].layer, Seed[]), lemmas[L].seed)
            end
            if n.kind == "c"
                Zin, Z, _, ok = propagate(N, lo, hi, active, cache, seeds, adapter_stop)
                if o.kind == "main"
                    sat, _, _, _, db = epsprop(X1, X2, Zin, Z, nothing)
                    verified_leaf[id] = sat && ok && !isnan(db)
                else
                    s = lem_self.seed
                    b = zono_bounds(Z.∂Z)
                    good = all(b[s.neurons, 1] .>= s.dlo) && all(b[s.neurons, 2] .<= s.dhi)
                    if !isnothing(s.n1); b1 = zono_bounds(Z.Z₁); good &= all(b1[s.neurons, 1] .>= s.n1[1]) && all(b1[s.neurons, 2] .<= s.n1[2]); end
                    if !isnothing(s.n2); b2 = zono_bounds(Z.Z₂); good &= all(b2[s.neurons, 1] .>= s.n2[1]) && all(b2[s.neurons, 2] .<= s.n2[2]); end
                    verified_leaf[id] = good && ok
                end
                verbose && println(stderr, "leaf $id: ", verified_leaf[id])
                continue
            elseif n.kind == "o"
                verified_leaf[id] = false
                continue
            end
            out_cache = cache
            if n.p["p"] == true
                _, _, out_cache, _ = propagate(N, lo, hi, active, cache, seeds, adapter_stop)
            end
            if n.kind == "s"
                d = Int(n.p["d"]); c = Float64(n.p["c"])
                need(d in active, "split node $id on inactive dim $d")
                need(lo[d] < c < hi[d], "split node $id: cut $c not strictly inside ($(lo[d]), $(hi[d]))")
                for ch in children[id]
                    l2 = copy(lo); h2 = copy(hi)
                    nodes[ch].side == 1 ? (h2[d] = c) : (l2[d] = c)
                    region[ch] = (l2, h2)
                    push!(stack, (ch, deepcopy(out_cache), copy(scope)))
                end
            else # assume
                L = Int(n.p["l"])
                ch = children[id][1]
                region[ch] = (copy(lo), copy(hi))
                push!(stack, (ch, deepcopy(out_cache), vcat(scope, [L])))
                # the lemma's proof root region is this node's region
                lr = obls[lemmas[L].obligation].root
                region[lr] = (copy(lo), copy(hi))
            end
        end
    end
    replay(mains[1])
    done = Set([mains[1].id])
    progress = true
    while progress
        progress = false
        for o in values(obls)
            o.id in done && continue
            if haskey(region, o.root)
                replay(o); push!(done, o.id); progress = true
            end
        end
    end
    need(length(done) == length(obls), "unreachable obligations (lemmas whose assume node is not in the tree)")
    need(length(obl_of) == length(nodes), "nodes not reachable from any obligation root")

    # closure (with lemma cycle detection)
    memo = Dict{Int,Bool}(); lmemo = Dict{Int,Bool}(); onstack = Set{Int}()
    function closed(id)
        haskey(memo, id) && return memo[id]
        n = nodes[id]
        v = n.kind == "c" ? verified_leaf[id] : n.kind == "o" ? false :
            n.kind == "s" ? all(closed, children[id]) : (closed(children[id][1]) && proven(Int(n.p["l"])))
        memo[id] = v
        return v
    end
    function proven(L)
        haskey(lmemo, L) && return lmemo[L]
        L in onstack && reject("circular lemma dependency at lemma $L")
        push!(onstack, L)
        v = closed(obls[lemmas[L].obligation].root)
        delete!(onstack, L)
        lmemo[L] = v
        return v
    end
    for l in values(C["lemmas"])
        if l["status"] == "proven"
            need(proven(Int(l["id"])), "lemma $(l["id"]) claimed proven but its proof does not replay")
        end
    end
    failed = [id for (id, n) in nodes if n.kind == "c" && !verified_leaf[id]]
    need(isempty(failed), "closed leaves that do not re-check: $(failed[1:min(5, end)])")
    mroot = mains[1].root
    if status == "SAFE"
        need(closed(mroot), "main obligation is not closed")
        return "ACCEPT SAFE nodes=$(length(nodes)) lemmas=$(length(lemmas))"
    end
    # PARTIAL: report verified closed volume (maximal closed subtrees of the main tree)
    vol = 0.0
    st = [(mroot, 1.0)]
    while !isempty(st)
        id, w = pop!(st)
        if closed(id)
            vol += w
        elseif nodes[id].kind == "s"
            for c in children[id]; push!(st, (c, w / 2)); end
        end
    end
    return "ACCEPT PARTIAL verified_closed_volume=$(round(vol; digits=6))"
end

function main(args)
    if length(args) < 4
        println(stderr, "usage: check_certificate.jl CERT NET1 NET2 SPEC [--verbose]")
        return 2
    end
    try
        r = check(args[1], args[2], args[3], args[4]; verbose="--verbose" in args)
        println(r)
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
