# Newline-delimited JSON-RPC over stdio.
#   request : {"id": <any>, "method": "<name>", "params": {...}}
#   response: {"id": <same>, "result": ...}  or  {"id": <same>, "error": {"type": ..., "message": ...}}
# Julia's own stdout (VeryDiff prints progress with println) is redirected to stderr so that the
# protocol channel stays clean.

opt(p, k, default) = (v = get(p, k, nothing); isnothing(v) ? default : v)
fvec(v) = Float64[decode_float(x) for x in v]
ivec(v) = isnothing(v) ? nothing : Int[Int(x) for x in v]

"""Exercise the main code paths on a bundled tiny instance to trigger JIT compilation."""
function warmup()
    t = time()
    w = joinpath(@__DIR__, "..", "warmup")
    S = open_session(joinpath(w, "net1.onnx"), joinpath(w, "net2.onnx"), joinpath(w, "spec.vnnlib"),
                     Dict("kind" => "epsilon", "epsilon" => 1e-6))
    auto_expand!(S, 1; max_nodes=6)
    leaf = list_open_leaves(S)["leaves"][1]["id"]
    k = S.relu_layers[1]
    r = introduce_lemma!(S, leaf, k, [1], [-10.0], [10.0])
    prove_lemma!(S, r["lemma"]; max_nodes=2)
    check!(S, r["child"])
    session_status(S); list_layers(S); read_weights(S, 1); get_node(S, 1); output_bounds(S, leaf); split_scores(S, leaf)
    inspect_layer(S, leaf, k; detail="generators_topk", neurons=[1]); cache_bounds(S, leaf, k)
    eval_point(S, (S.root_box.lo .+ S.root_box.hi) ./ 2)
    sanitize(certificate(S))
    retract_lemma!(S, r["lemma"])
    prune!(S, 1)
    SESSION[] = nothing
    return Dict("warmup_seconds" => round(time() - t; digits=2))
end

"""Optional per-search node cap (env VDA_MAX_AUTO_NODES; used by the harness's min-nodes objective)."""
auto_cap() = parse(Int, get(ENV, "VDA_MAX_AUTO_NODES", "1000000000"))

const METHODS = Dict{String,Function}(
    "warmup" => p -> warmup(),
    "ping" => p -> Dict("pong" => true, "kernel_version" => KERNEL_VERSION),
    "open_session" => p -> begin
        S = open_session(String(p["net1"]), String(p["net2"]), String(p["spec"]), p["property"])
        Dict("spec_hash" => S.spec_hash, "root" => S.obligations[S.main_obligation].root,
             "input_dim" => dim(S.root_box), "active_dims" => S.active_dims, "relu_layers" => S.relu_layers,
             "softmax_removed" => S.softmax_removed, "verydiff_commit" => S.verydiff_commit)
    end,
    "check" => p -> check!(current(), Int(p["node"]); rationale=opt(p, "rationale", "")),
    "split" => p -> split!(current(), Int(p["node"]), Int(p["dim"]); cut=get(p, "cut", nothing),
                           cuts=get(p, "cuts", nothing), dim_block=opt(p, "dim_block", "primary"), rationale=opt(p, "rationale", "")),
    "auto_expand" => p -> isnothing(get(p, "per_leaf_max_nodes", nothing)) ?
                           auto_expand!(current(), Int(p["node"]); max_nodes=min(Int(opt(p, "max_nodes", 1000)), auto_cap()),
                           max_seconds=Float64(opt(p, "max_seconds", 60.0)), heuristic=opt(p, "heuristic", "verydiff"),
                           rationale=opt(p, "rationale", "")) :
                           auto_expand_per_leaf!(current(), Int(p["node"]); per_leaf_max_nodes=min(Int(p["per_leaf_max_nodes"]), auto_cap()),
                           max_nodes=Int(opt(p, "max_nodes", 100_000)), max_seconds=Float64(opt(p, "max_seconds", 60.0)),
                           heuristic=opt(p, "heuristic", "verydiff"), rationale=opt(p, "rationale", "")),
    "introduce_lemma" => p -> introduce_lemma!(current(), Int(p["node"]), Int(p["relu_layer"]), ivec(p["neurons"]),
                           p["dlo"], p["dhi"]; n1_bounds=get(p, "n1_bounds", nothing), n2_bounds=get(p, "n2_bounds", nothing),
                           rationale=opt(p, "rationale", "")),
    "introduce_phase_lemma" => p -> introduce_phase_lemma!(current(), Int(p["node"]), Int(p["relu_layer"]), ivec(p["neurons"]);
                           net1=opt(p, "net1", "any"), net2=opt(p, "net2", "any"), rationale=opt(p, "rationale", "")),
    "trial_lemma" => p -> trial_lemma!(current(), Int(p["node"]), Int(p["relu_layer"]), ivec(p["neurons"]);
                           kind=opt(p, "kind", "phase"), dlo=get(p, "dlo", nothing), dhi=get(p, "dhi", nothing),
                           n1_bounds=get(p, "n1_bounds", nothing), n2_bounds=get(p, "n2_bounds", nothing),
                           net1=opt(p, "net1", "any"), net2=opt(p, "net2", "any"),
                           lemma_max_nodes=min(Int(opt(p, "lemma_max_nodes", 2000)), 200_000),
                           main_max_nodes=min(Int(opt(p, "main_max_nodes", 2000)), 200_000),
                           max_seconds=Float64(opt(p, "max_seconds", 120.0)), baseline=Bool(opt(p, "baseline", true)),
                           rationale=opt(p, "rationale", "")),
    "prove_lemma" => p -> prove_lemma!(current(), Int(p["lemma"]); max_nodes=Int(opt(p, "max_nodes", 1000)),
                           max_seconds=Float64(opt(p, "max_seconds", 60.0)), rationale=opt(p, "rationale", "")),
    "retract_lemma" => p -> retract_lemma!(current(), Int(p["lemma"]); rationale=opt(p, "rationale", "")),
    "refute_lemma" => p -> refute_lemma!(current(), Int(p["lemma"]), fvec(p["x"]); rationale=opt(p, "rationale", "")),
    "prune" => p -> prune!(current(), Int(p["node"]); rationale=opt(p, "rationale", "")),
    "submit_counterexample" => p -> submit_counterexample!(current(), fvec(p["x"]); rationale=opt(p, "rationale", "")),
    "export_certificate" => p -> export_certificate(current(), String(p["path"])),
    "session_status" => p -> session_status(current()),
    "list_layers" => p -> list_layers(current()),
    "list_open_leaves" => p -> list_open_leaves(current(); sort_by=opt(p, "sort_by", "distance_bound"),
                           limit=Int(opt(p, "limit", 10)), offset=Int(opt(p, "offset", 0)),
                           obligation=get(p, "obligation", nothing), under=get(p, "under", nothing)),
    "get_node" => p -> get_node(current(), Int(p["node"])),
    "inspect_layer" => p -> inspect_layer(current(), Int(p["node"]), Int(p["layer"]); which=opt(p, "which", "d"),
                           neurons=ivec(get(p, "neurons", nothing)), detail=opt(p, "detail", "bounds"),
                           topk=Int(opt(p, "topk", 5)), limit=Int(opt(p, "limit", 16))),
    "output_bounds" => p -> output_bounds(current(), Int(p["node"]); neurons=ivec(get(p, "neurons", nothing))),
    "split_scores" => p -> split_scores(current(), Int(p["node"]); top_k=Int(opt(p, "top_k", 5))),
    "cache_bounds" => p -> cache_bounds(current(), Int(p["node"]), Int(p["relu_layer"]); neurons=ivec(get(p, "neurons", nothing))),
    "eval_point" => p -> begin
        r = eval_point(current(), fvec(p["x"]); hidden=Bool(opt(p, "hidden", false)))
        Dict(k => (v isa AbstractVector{<:Real} ? r6(v) : v isa Real ? r6(v) : v) for (k, v) in r)
    end,
    "read_weights" => p -> read_weights(current(), Int(p["layer"]); which=opt(p, "which", "diff"),
                           rows=ivec(get(p, "rows", nothing)), cols=ivec(get(p, "cols", nothing)),
                           max_entries=Int(opt(p, "max_entries", 400)), digits=Int(opt(p, "digits", 4))),
    "batch" => p -> run_batch(p["actions"]; stop_on_error=Bool(opt(p, "stop_on_error", true))),
    "export_data" => p -> export_data(current(), String(p["what"]), get(p, "node_ids", nothing), String(p["path"])),
)

"""Make a result JSON-safe: non-finite floats become strings, symbols become strings."""
sanitize(x::AbstractFloat) = isfinite(x) ? x : encode_float(x)
sanitize(x::Symbol) = string(x)
sanitize(x::AbstractDict) = Dict{String,Any}(string(k) => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = Any[sanitize(v) for v in x]
sanitize(x::Tuple) = Any[sanitize(v) for v in x]
sanitize(x) = x

# Actions allowed inside `batch` (no session/certificate/export side effects).
const BATCHABLE = Set(["check", "split", "auto_expand", "introduce_lemma", "introduce_phase_lemma", "trial_lemma", "prove_lemma", "retract_lemma",
    "refute_lemma", "prune", "submit_counterexample", "session_status", "list_open_leaves", "get_node",
    "output_bounds", "split_scores", "cache_bounds", "inspect_layer", "read_weights", "eval_point"])

"""Run several actions in one request, in order. Each action is checked by the kernel exactly as if it
were sent alone. `\$k.children[i]` in a node parameter refers to the i-th child returned by action k (1-based)."""
function run_batch(actions; stop_on_error::Bool=true)
    results = Any[]
    function resolve(v)
        if v isa AbstractString && startswith(v, "\$")
            m = match(r"^\$(\d+)\.children\[(\d+)\]$", v)
            isnothing(m) && kerr("bad_request", "bad reference '$v' (use \$k.children[i])")
            k, i = parse(Int, m[1]), parse(Int, m[2])
            (1 <= k <= length(results) && results[k] isa AbstractDict && haskey(results[k], "children")) ||
                kerr("bad_request", "reference '$v' points to an action without children")
            ch = results[k]["children"]
            1 <= i <= length(ch) || kerr("bad_request",
                "reference '$v': child index must be in 1:$(length(ch)) (children are 1-based)")
            return ch[i]
        end
        return v
    end
    for (j, a) in enumerate(actions)
        m = string(get(a, "method", ""))
        if !(m in BATCHABLE)
            push!(results, Dict("error" => Dict("type" => "bad_request", "message" => "'$m' not allowed in batch")))
            stop_on_error && break
            continue
        end
        try
            params = Dict{String,Any}(string(k) => resolve(v) for (k, v) in get(a, "params", Dict()))
            push!(results, METHODS[m](params))
        catch e
            e isa InterruptException && rethrow()
            push!(results, Dict("error" => error_payload(e)))
            stop_on_error && break
        end
    end
    return Dict("n_actions" => length(actions), "n_done" => length(results), "results" => results,
                "verdict" => session_verdict(current()))
end

function error_payload(e)
    if e isa KernelError
        return Dict("type" => e.type, "message" => e.msg)
    elseif e isa NotImplementedError
        return Dict("type" => "not_implemented", "message" => e.msg)
    elseif e isa KeyError
        return Dict("type" => "bad_request", "message" => "missing parameter or key: $(e.key)")
    else
        return Dict("type" => "internal", "message" => sprint(showerror, e)[1:min(end, 2000)])
    end
end

function handle(line::AbstractString)
    req = try JSON.parse(line) catch e
        return Dict("id" => nothing, "error" => Dict("type" => "bad_request", "message" => "invalid JSON"))
    end
    id = get(req, "id", nothing)
    m = string(get(req, "method", ""))
    params = get(req, "params", Dict{String,Any}())
    isnothing(params) && (params = Dict{String,Any}())
    haskey(METHODS, m) || return Dict("id" => id, "error" => Dict("type" => "bad_request", "message" => "unknown method '$m'"))
    try
        return Dict("id" => id, "result" => sanitize(METHODS[m](params)))
    catch e
        e isa InterruptException && rethrow()
        @error "RPC method failed" m exception = (e, catch_backtrace())
        return Dict("id" => id, "error" => error_payload(e))
    end
end

"""Serve requests on stdin/stdout until EOF or a `shutdown` request."""
function serve(input::IO=stdin, output::IO=stdout)
    out = output
    redirect_stdout(stderr)
    println(out, JSON.json(Dict("id" => nothing, "result" => Dict("ready" => true, "kernel_version" => KERNEL_VERSION))))
    flush(out)
    while !eof(input)
        line = readline(input)
        isempty(strip(line)) && continue
        if occursin("\"shutdown\"", line)
            req = try JSON.parse(line) catch; nothing end
            if !isnothing(req) && get(req, "method", "") == "shutdown"
                println(out, JSON.json(Dict("id" => get(req, "id", nothing), "result" => Dict("bye" => true))))
                flush(out)
                break
            end
        end
        resp = handle(line)
        println(out, JSON.json(resp))
        flush(out)
    end
end
