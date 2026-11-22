# Dry-run planning primitive: evaluate candidate expansions of a node without changing the proof.

"""Local snapshot: a trial only adds nodes below `node` and changes `node` itself (no lemmas)."""
local_snapshot(S::Session, node::Int) = (deepcopy(S.nodes[node]), S.next_node, length(S.log), S.n_propagations, S.n_checks, S.cex)
function local_restore!(S::Session, node::Int, s)
    discard_below!(S, S.nodes[node])
    for id in collect(keys(S.nodes))
        id >= s[2] && delete!(S.nodes, id)      # anything created during the trial
    end
    S.nodes[node] = s[1]
    S.next_node = s[2]
    resize!(S.log, s[3])
    used = S.n_propagations - s[4]
    S.n_propagations, S.n_checks, S.cex = s[4], s[5], s[6]
    S.n_trial_propagations += used
    return used
end

"""
apply_action!(S, node, a) applies one expansion `a` to the open leaf `node` and returns the new leaves:
  {"kind": "split", "dim": d, "cut": c (optional)}                       input split
  {"kind": "split_relu", "layer": k, "neuron": j, "var": "z1|z2|d", "t": 0}  case split on a pre-activation
  {"kind": "phase4", "layer": k, "neuron": j}                              z1 and z2 phase split (4 leaves)
"""
function apply_action!(S::Session, node::Int, a::AbstractDict)
    kind = String(a["kind"])
    if kind == "split"
        r = split!(S, node, Int(a["dim"]); cut=get(a, "cut", nothing))
        return Int.(r["children"])
    elseif kind == "split_relu"
        r = split_relu!(S, node, Int(a["layer"]), Int(a["neuron"]); var=String(get(a, "var", "z1")), t=Float64(get(a, "t", 0.0)))
        return Int.(r["children"])
    elseif kind == "phase4"
        k, j = Int(a["layer"]), Int(a["neuron"])
        out = Int[]
        for c in split_relu!(S, node, k, j; var="z1")["children"]
            append!(out, split_relu!(S, c, k, j; var="z2")["children"])
        end
        return out
    else
        kerr("precondition", "unknown action kind '$kind' (split | split_relu | phase4)")
    end
end

"""
trial_actions!(S, node, actions; max_nodes) — dry run: for each candidate expansion of `node`, apply it and
finish all resulting leaves with VeryDiff's best-first search (at most `max_nodes` checks), record the number
of checks needed (= propagations that the finished subtree would put in the certificate), then restore the
session. `node` itself counts 1 if it has not been checked yet. "baseline" is plain auto_expand from `node`.
Proof state is unchanged; the cost is reported as trial propagations.
"""
function trial_actions!(S::Session, node::Int, actions::AbstractVector; max_nodes::Int=2000, baseline::Bool=true,
                        max_seconds::Float64=600.0)
    n = getnode(S, node)
    require_open_leaf(n)
    isempty(S.lemmas) || kerr("precondition", "trial_actions requires a session without lemmas (local snapshot)")
    out = Dict{String,Any}[]
    own = isnothing(n.last_check) ? 1 : 0
    cands = Any[]
    baseline && push!(cands, nothing)
    append!(cands, actions)
    for a in cands
        snap = local_snapshot(S, node)
        c0 = S.n_checks
        r = Dict{String,Any}("action" => isnothing(a) ? "baseline" : a)
        try
            if isnothing(a)
                st = auto_expand!(S, node; max_nodes=max_nodes, max_seconds=max_seconds, log=false)
            else
                isnothing(S.nodes[node].last_check) && check!(S, node)
                if S.nodes[node].kind == :open
                    apply_action!(S, node, a)
                    st = auto_expand!(S, node; max_nodes=max_nodes, max_seconds=max_seconds, log=false)
                end
            end
            r["closed"] = is_closed(S, node)
            r["checks"] = S.n_checks - c0
            r["cex"] = !isnothing(S.cex)
        catch e
            e isa KernelError || rethrow()
            r["error"] = e.msg
        end
        local_restore!(S, node, snap)
        push!(out, r)
    end
    return Dict("node" => node, "max_nodes" => max_nodes, "results" => out,
                "note" => "checks include the node's own check if it was unchecked; closed=false means the cap was hit")
end
