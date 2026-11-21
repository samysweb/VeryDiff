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
    session_status(S); list_layers(S); get_node(S, 1); output_bounds(S, leaf); split_scores(S, leaf)
    inspect_layer(S, leaf, k; detail="generators_topk", neurons=[1]); cache_bounds(S, leaf, k)
    eval_point(S, (S.root_box.lo .+ S.root_box.hi) ./ 2)
    sanitize(certificate(S))
    retract_lemma!(S, r["lemma"])
    prune!(S, 1)
    SESSION[] = nothing
    return Dict("warmup_seconds" => round(time() - t; digits=2))
end

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
    "auto_expand" => p -> auto_expand!(current(), Int(p["node"]); max_nodes=Int(opt(p, "max_nodes", 1000)),
                           max_seconds=Float64(opt(p, "max_seconds", 60.0)), heuristic=opt(p, "heuristic", "verydiff"),
                           rationale=opt(p, "rationale", "")),
    "introduce_lemma" => p -> introduce_lemma!(current(), Int(p["node"]), Int(p["relu_layer"]), ivec(p["neurons"]),
                           p["dlo"], p["dhi"]; n1_bounds=get(p, "n1_bounds", nothing), n2_bounds=get(p, "n2_bounds", nothing),
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
    "export_data" => p -> export_data(current(), String(p["what"]), get(p, "node_ids", nothing), String(p["path"])),
)

"""Make a result JSON-safe: non-finite floats become strings, symbols become strings."""
sanitize(x::AbstractFloat) = isfinite(x) ? x : encode_float(x)
sanitize(x::Symbol) = string(x)
sanitize(x::AbstractDict) = Dict{String,Any}(string(k) => sanitize(v) for (k, v) in x)
sanitize(x::AbstractVector) = Any[sanitize(v) for v in x]
sanitize(x::Tuple) = Any[sanitize(v) for v in x]
sanitize(x) = x

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
