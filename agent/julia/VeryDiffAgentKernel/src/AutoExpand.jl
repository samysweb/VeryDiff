# auto_expand: VeryDiff's branch-and-bound loop (mirrors Verifier.jl::worker_function_internal)
# on a subtree, but every check and every split is recorded as a tree node (created_by = :auto).

struct QItem
    db::Float64
    id::Int
end
Base.isless(a::QItem, b::QItem) = a.db < b.db || (a.db == b.db && a.id > b.id)

function open_leaves_below(S::Session, id::Int)
    return [x for x in subtree(S, id) if S.nodes[x].kind == :open]
end

"""Pick a split dimension for the just-checked node `n` (flat index → input dimension)."""
function choose_dim(S::Session, n::Node, res::PropResult, heur, heuristic::Symbol)
    a = obligation_of(S, n).adapter
    if heuristic == :verydiff
        flat = adapter_split_heuristic(a, res.Zin, res.Zout, heur, res.task)
        return res.task.distance_indices[flat]
    elseif heuristic == :widest
        w = [(n.region.hi[i] - n.region.lo[i]) / (S.root_box.hi[i] - S.root_box.lo[i]) for i in S.active_dims]
        return S.active_dims[argmax(w)]
    else
        kerr("precondition", "unknown heuristic $heuristic (use verydiff | widest)")
    end
end

function auto_expand!(S::Session, node::Int; max_nodes::Int=1000, max_seconds::Float64=60.0,
                      heuristic::AbstractString="verydiff", rationale::AbstractString="", log::Bool=true)
    timed(S) do
        getnode(S, node)
        log && log_action!(S, "auto_expand", Dict("node" => node, "max_nodes" => max_nodes,
            "max_seconds" => max_seconds, "heuristic" => heuristic), rationale)
        h = Symbol(heuristic)
        t0 = time()
        heap = BinaryMaxHeap{QItem}()
        for x in open_leaves_below(S, node)
            push!(heap, QItem(S.nodes[x].distance_bound, x))
        end
        checked = 0; closed = 0; splits = 0; stuck = Int[]
        stop_reason = "exhausted"
        while !isempty(heap)
            if checked >= max_nodes
                stop_reason = "max_nodes"; break
            end
            if time() - t0 > max_seconds
                stop_reason = "max_seconds"; break
            end
            it = pop!(heap)
            n = S.nodes[it.id]
            outcome, res, heur = do_check!(S, n)
            checked += 1
            if outcome == :closed
                closed += 1
                continue
            elseif outcome == :cex
                stop_reason = "counterexample"; break
            elseif outcome == :refuted
                stop_reason = "lemma_refuted"; break
            end
            d = choose_dim(S, n, res, heur, h)
            lo, hi = n.region.lo[d], n.region.hi[d]
            if !(lo < lo / 2 + hi / 2 < hi)
                push!(stuck, n.id)          # cannot bisect further in Float64
                continue
            end
            c1, c2 = do_split!(S, n, d, nothing; created_by=:auto)
            splits += 1
            push!(heap, QItem(c1.distance_bound, c1.id))
            push!(heap, QItem(c2.distance_bound, c2.id))
        end
        leaves = open_leaves_below(S, node)
        dbs = sort([S.nodes[x].distance_bound for x in leaves]; rev=true)
        return Dict{String,Any}("node" => node, "checked" => checked, "closed" => closed, "splits" => splits,
            "open_leaves" => length(leaves), "stuck_leaves" => stuck, "stop_reason" => stop_reason,
            "worst_distance_bounds" => dbs[1:min(5, length(dbs))],
            "subtree_closed" => is_closed(S, node), "seconds" => round(time() - t0; digits=3),
            "verdict" => session_verdict(S))
    end
end

"""Per-leaf mode: run an independent best-first search (budget `per_leaf_max_nodes`) on every open leaf
under `node` that exists when the call starts, until the global `max_nodes`/`max_seconds` budget is used.
Equivalent to calling auto_expand on each branch, in one action."""
function auto_expand_per_leaf!(S::Session, node::Int; per_leaf_max_nodes::Int=100, max_nodes::Int=100_000,
                               max_seconds::Float64=60.0, heuristic::AbstractString="verydiff",
                               rationale::AbstractString="")
    getnode(S, node)
    log_action!(S, "auto_expand", Dict("node" => node, "per_leaf_max_nodes" => per_leaf_max_nodes,
        "max_nodes" => max_nodes, "max_seconds" => max_seconds, "heuristic" => heuristic), rationale)
    t0 = time()
    leaves = open_leaves_below(S, node)
    used = 0
    per = Dict{String,Any}[]
    for x in leaves
        remaining = max_nodes - used
        tleft = max_seconds - (time() - t0)
        (remaining <= 0 || tleft <= 0) && break
        haskey(S.nodes, x) && S.nodes[x].kind == :open || continue
        r = auto_expand!(S, x; max_nodes=min(per_leaf_max_nodes, remaining), max_seconds=tleft,
                         heuristic=heuristic, log=false)
        used += r["checked"]
        push!(per, Dict("leaf" => x, "checked" => r["checked"], "closed" => r["subtree_closed"], "open_leaves" => r["open_leaves"]))
        r["stop_reason"] in ("counterexample", "lemma_refuted") && break
    end
    rem = open_leaves_below(S, node)
    dbs = sort([S.nodes[i].distance_bound for i in rem]; rev=true)
    return Dict{String,Any}("node" => node, "leaves_processed" => length(per), "leaves_at_start" => length(leaves),
        "checked" => used, "leaves_closed" => count(d -> d["closed"], per),
        "unclosed_leaves" => [d["leaf"] for d in per if !d["closed"]][1:min(end, 30)],
        "open_leaves" => length(rem), "worst_distance_bounds" => dbs[1:min(5, end)],
        "subtree_closed" => is_closed(S, node), "seconds" => round(time() - t0; digits=3), "verdict" => session_verdict(S))
end
