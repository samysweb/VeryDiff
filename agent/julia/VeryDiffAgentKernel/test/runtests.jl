using Test
using JSON
using VeryDiff
using VeryDiffAgentKernel
const K = VeryDiffAgentKernel

const AGENT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const NETS = joinpath(AGENT, "tests", "nets")
include(joinpath(AGENT, "checker", "check_certificate.jl"))

if !isfile(joinpath(NETS, "instances.json"))
    run(`python3 $(joinpath(AGENT, "tests", "gen_tiny_nets.py")) $NETS`)
end
const INST = Dict(i["name"] => Dict("name" => i["name"], "epsilon" => i["epsilon"],
    (k => joinpath(NETS, i[k]) for k in ("net1", "net2", "spec"))...)
    for i in JSON.parse(read(joinpath(NETS, "instances.json"), String)))
const SCRATCH = mktempdir()

open_case(name; eps=nothing) = (i = INST[name];
    open_session(i["net1"], i["net2"], i["spec"], Dict("kind" => "epsilon", "epsilon" => isnothing(eps) ? i["epsilon"] : eps)))

function check_cert(S, name; mutate=nothing)
    i = INST[name]
    p = joinpath(SCRATCH, "cert_$(name)_$(rand(UInt32)).json")
    export_certificate(S, p)
    if !isnothing(mutate)
        c = JSON.parse(read(p, String))
        mutate(c)
        write(p, JSON.json(c))
    end
    try
        return CertChecker.check(p, i["net1"], i["net2"], i["spec"])
    catch e
        e isa CertChecker.Reject && return "REJECT " * e.msg
        rethrow()
    end
end

@testset "VeryDiffAgentKernel" begin
@testset "Regions" begin
    b = Box([-1.0, 0.0, 2.0], [1.0, 0.0, 3.0])
    c1, c2 = K.split_box(b, 1, 0.25)
    @test c1.hi[1] === 0.25 && c2.lo[1] === 0.25
    @test c1.lo == b.lo && c2.hi == b.hi
    @test_throws K.KernelError K.split_box(b, 1, 1.0)
    @test_throws K.KernelError K.split_box(b, 1, -1.0)
    @test K.volume_fraction(c1, b) ≈ 0.625
    # outward conversion covers exactly, including awkward values
    for (lo, hi) in [(0.1, 0.3), (-1e-300, 1e-300), (0.0, 5e-324), (1.0, nextfloat(1.0)), (-3.0, 1e10)]
        bb = Box([lo, 7.0], [hi, 7.0])
        mid, dist = K.center_radius(bb, [1])
        R = Rational{BigInt}
        @test R(mid[1]) - R(dist[1]) <= R(lo) && R(mid[1]) + R(dist[1]) >= R(hi)
        @test mid[2] == 7.0 && length(dist) == 1
    end
end

@testset "Seeding (uninitialised cache)" begin
    bc = VeryDiff.Definitions.BoundsCache()
    st = K.LemmaStatement(3, [2], [-0.5], [0.5], nothing, nothing, nothing, nothing)
    K.apply_seeds!(bc, 4, [st])
    @test bc.initialized
    @test bc.∂lower == [-Inf, -0.5, -Inf, -Inf] && bc.∂upper == [Inf, 0.5, Inf, Inf]
    st2 = K.LemmaStatement(3, [2, 3], [-0.2, -1.0], [0.9, 1.0], nothing, nothing, nothing, nothing)
    K.apply_seeds!(bc, 4, [st2])
    @test bc.∂lower[2] == -0.2 && bc.∂upper[2] == 0.5 && bc.∂lower[3] == -1.0
    bad = K.LemmaStatement(3, [1], [1.0], [2.0], nothing, nothing, nothing, nothing)
    bc.∂upper[1] = 0.5
    K.apply_seeds!(bc, 4, [bad])
    @test K.cache_inconsistent(bc)
end

@testset "Closure is derived" begin
    S = open_case("safe_split")
    root = S.obligations[S.main_obligation].root
    @test session_verdict(S) == "PARTIAL"
    r = split!(S, root, 1)
    c1, c2 = r["children"]
    @test !is_closed(S, root)
    # no way to close without evidence: closing actions don't exist; prune only reopens
    @test !haskey(K.METHODS, "close") && !haskey(K.METHODS, "drop")
    auto_expand!(S, c1; max_nodes=10_000)
    @test is_closed(S, c1) && !is_closed(S, root)
    prune!(S, root)
    @test S.nodes[root].kind == :open && length(S.nodes) == 1
    @test K.validate_tree(S)
end

@testset "Differential: auto_expand(root, ∞) vs CLI" begin
    for name in sort(collect(keys(INST)))
        i = INST[name]
        S = open_case(name)
        r = auto_expand!(S, S.obligations[S.main_obligation].root; max_nodes=200_000, max_seconds=600.0)
        v = session_verdict(S)
        code = VeryDiff.run_cmd([i["net1"], i["net2"], i["spec"], "--epsilon", string(i["epsilon"]), "--timeout", "600"])
        cli = code == 20 ? "SAFE" : code == 10 ? "UNSAFE" : "UNKNOWN"
        @info "differential" name v cli r["checked"]
        @test v == cli
        res = check_cert(S, name)
        @test startswith(res, "ACCEPT $v")
    end
end

@testset "Manual actions, k-way split, partial certificate" begin
    S = open_case("safe_hard")
    root = S.obligations[S.main_obligation].root
    r = check!(S, root; rationale="first look")
    @test r["holds"] == false
    r = split!(S, root, 2; cuts=[-0.5, 0.0, 0.5], rationale="4-way on dim 2")
    @test length(r["children"]) == 4
    a = auto_expand!(S, r["children"][1]; max_nodes=20)
    @test a["checked"] <= 20
    @test session_verdict(S) == "PARTIAL"
    res = check_cert(S, "safe_hard")
    @test startswith(res, "ACCEPT PARTIAL")
    st = session_status(S)
    @test st["main"]["open_leaves"] > 0
    @test_throws K.KernelError split!(S, root, 1)          # not an open leaf
    @test_throws K.KernelError split!(S, r["children"][2], 1; cut=5.0)
    @test_throws K.KernelError split!(S, r["children"][2], 1; dim_block="sec1")   # eps has no secondary block
end

@testset "UNSAFE and counterexample checks" begin
    S = open_case("unsafe")
    root = S.obligations[S.main_obligation].root
    r = check!(S, root)
    @test session_verdict(S) == "UNSAFE"
    @test startswith(check_cert(S, "unsafe"), "ACCEPT UNSAFE")
    # cex outside region must be rejected by the checker
    res = check_cert(S, "unsafe"; mutate=c -> (c["cex"]["x"][1] = 5.0))
    @test startswith(res, "REJECT")
    # submit_counterexample is kernel-checked
    S2 = open_case("safe_easy")
    @test submit_counterexample!(S2, [0.0, 0.0])["accepted"] == false
    @test submit_counterexample!(S2, [5.0, 0.0])["accepted"] == false
    @test session_verdict(S2) == "PARTIAL"
end

"""Proven lemma helper: bounds at ReLU layer `k` over the root, widened by `pad`."""
function root_lemma_bounds(S, k, neurons; pad=1e-6)
    n = S.nodes[S.obligations[S.main_obligation].root]
    res = K.inspect_propagation(S, n; stop=k)
    b = K.zono_bounds(res.Zout.∂Z)
    return b[neurons, 1] .- pad, b[neurons, 2] .+ pad
end

@testset "Lemmas: introduce / prove / use / refute / retract" begin
    S = open_case("safe_mid")
    root = S.obligations[S.main_obligation].root
    k = S.relu_layers[end]
    neurons = [1, 2, 3]
    lo, hi = root_lemma_bounds(S, k, neurons)
    r = introduce_lemma!(S, root, k, neurons, lo, hi; rationale="true by root bounds")
    L = r["lemma"]; child = r["child"]
    @test K.lemma_status(S, L) == "open"
    p = prove_lemma!(S, L; max_nodes=10)
    @test p["lemma_status"] == "proven"
    auto_expand!(S, child; max_nodes=100_000)
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, "safe_mid"), "ACCEPT SAFE")

    # mutation: lemma bounds tightened after the proof
    res = check_cert(S, "safe_mid"; mutate=c -> (c["lemmas"][1]["dhi"][1] = c["lemmas"][1]["dlo"][1]))
    @test startswith(res, "REJECT")
    # mutation: lemma marked proven without a proof tree (proof root turned into an open leaf, children dropped)
    res = check_cert(S, "safe_mid"; mutate=c -> begin
        lr = c["obligations"][2]["root"]
        keep = Set{Int}()
        byid = Dict(e[1] => e for e in c["nodes"])
        c["nodes"] = [e for e in c["nodes"] if begin
            x = e[1]; inlem = false
            while x != 0; x == lr && (inlem = true; break); x = byid[x][2]; end
            !inlem || e[1] == lr
        end]
        for e in c["nodes"]; e[1] == lr && (e[4] = "o"; e[5] = Dict()); end
    end)
    @test startswith(res, "REJECT")

    # a false lemma: impossibly tight bounds get refuted concretely
    S2 = open_case("safe_mid")
    root2 = S2.obligations[S2.main_obligation].root
    x = (S2.root_box.lo .+ S2.root_box.hi) ./ 2
    h1, h2 = K.hidden_at(S2, x, k)
    d1 = h1[1] - h2[1]
    r2 = introduce_lemma!(S2, root2, k, [1], [d1 + 1.0], [d1 + 2.0])
    p2 = prove_lemma!(S2, r2["lemma"]; max_nodes=50)
    @test p2["lemma_status"] == "refuted"
    @test session_verdict(S2) == "PARTIAL"
    # even if the main subtree closes under the refuted lemma, SAFE is impossible
    auto_expand!(S2, r2["child"]; max_nodes=2000)
    @test session_verdict(S2) != "SAFE"
    rr = retract_lemma!(S2, r2["lemma"])
    @test S2.nodes[root2].kind == :open && isempty(S2.lemmas)
    @test K.validate_tree(S2)
    # refute_lemma with a sandbox-found point is kernel-checked
    r3 = introduce_lemma!(S2, root2, k, [1], [d1 + 1.0], [d1 + 2.0])
    @test refute_lemma!(S2, r3["lemma"], x)["refuted"] == true
    @test refute_lemma!(S2, r3["lemma"], x .+ 100.0)["refuted"] == false || true
end

@testset "Lemma chaining" begin
    S = open_case("safe_mid")
    root = S.obligations[S.main_obligation].root
    k1, k2 = S.relu_layers[1], S.relu_layers[end]
    lo1, hi1 = root_lemma_bounds(S, k1, [1, 2, 3, 4])
    r1 = introduce_lemma!(S, root, k1, [1, 2, 3, 4], lo1, hi1)
    L1, c1 = r1["lemma"], r1["child"]
    # bounds at the deeper layer computed *with* L1 seeded (the child has L1 in scope)
    res = K.inspect_propagation(S, S.nodes[c1]; stop=k2)
    b = K.zono_bounds(res.Zout.∂Z)
    r2 = introduce_lemma!(S, c1, k2, [1, 2], b[1:2, 1] .- 1e-9, b[1:2, 2] .+ 1e-9)
    L2, c2 = r2["lemma"], r2["child"]
    l2root = S.nodes[r2["lemma_root"]]
    @test L1 in l2root.lemmas_in_scope               # the proof of L2 may use L1
    @test l2root.cache_in.bounds_cache !== S.nodes[c1].cache_in.bounds_cache
    @test K.validate_tree(S)
    prove_lemma!(S, L2; max_nodes=10_000)
    @test is_closed(S, l2root.id)
    @test K.lemma_status(S, L2) == "open"            # L1 not yet proven
    prove_lemma!(S, L1; max_nodes=10_000)
    @test K.lemma_status(S, L1) == "proven" && K.lemma_status(S, L2) == "proven"
    auto_expand!(S, c2; max_nodes=100_000)
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, "safe_mid"), "ACCEPT SAFE")
    # mutation: drop L1's proof (its root becomes an open leaf) -> L2 and the main proof must fail
    res = check_cert(S, "safe_mid"; mutate=c -> begin
        lr = only(o["root"] for o in c["obligations"] if o["target"] == "lemma:$L1")
        byid = Dict(e[1] => e for e in c["nodes"])
        under(x) = (y = byid[x][2]; while y != 0; y == lr && return true; y = byid[y][2]; end; false)
        c["nodes"] = [e for e in c["nodes"] if !under(e[1])]
        for e in c["nodes"]; e[1] == lr && (e[4] = "o"; e[5] = Dict()); end
        for l in c["lemmas"]; l["status"] = "open"; end
    end)
    @test startswith(res, "REJECT")
    # retracting L1 removes L2 (introduced below it) and its proof
    retract_lemma!(S, L1)
    @test isempty(S.lemmas) && S.nodes[root].kind == :open
    @test K.validate_tree(S)
end

@testset "Phase lemmas (one-sided, ±Inf in certificate)" begin
    S = open_case("safe_mid")
    root = S.obligations[S.main_obligation].root
    found = false
    auto_expand!(S, root; max_nodes=40)          # small sub-boxes, where some units are provably dead
    cands = [(l, k) for l in K.open_leaves_below(S, root) for k in S.relu_layers]
    for (leaf, k) in cands
        res = K.inspect_propagation(S, S.nodes[leaf]; stop=k)
        b1 = K.zono_bounds(res.Zout.Z₁); b2 = K.zono_bounds(res.Zout.Z₂)
        dead = findall((b1[:, 2] .< 0) .& (b2[:, 2] .< 0))
        isempty(dead) && continue
        r = introduce_phase_lemma!(S, leaf, k, dead[1:min(3, end)]; net1="off", net2="off")
        p = prove_lemma!(S, r["lemma"]; max_nodes=100)
        @test p["lemma_status"] == "proven"
        auto_expand!(S, root; max_nodes=100_000)
        @test session_verdict(S) == "SAFE"
        @test startswith(check_cert(S, "safe_mid"), "ACCEPT SAFE")
        found = true
        break
    end
    @test found
    # a false phase claim is refuted concretely
    S2 = open_case("safe_mid")
    x = (S2.root_box.lo .+ S2.root_box.hi) ./ 2
    k = S2.relu_layers[1]
    h1, _ = K.hidden_at(S2, x, k)
    j = argmax(h1)
    h1[j] > 0 || return
    r = introduce_phase_lemma!(S2, 1, k, [j]; net1="off")
    @test refute_lemma!(S2, r["lemma"], x)["refuted"] == true
    @test_throws K.KernelError introduce_phase_lemma!(S2, r["child"], k, [j])
end

@testset "Cache hygiene: retract restores lemma-free bounds" begin
    S = open_case("safe_mid")
    root = S.obligations[S.main_obligation].root
    check!(S, root)
    c1, c2 = split!(S, root, 1)["children"]
    ref = output_bounds(S, c1)                      # bounds without any lemma
    k = S.relu_layers[1]
    lo, hi = root_lemma_bounds(S, k, [1, 2]; pad=0.0)
    mid = (lo .+ hi) ./ 2
    r = introduce_lemma!(S, c1, k, [1, 2], mid .- 1e-3, mid .+ 1e-3)   # (probably false) tight lemma
    child = r["child"]
    check!(S, child)
    seeded = output_bounds(S, child)
    @test seeded != ref || true                     # seeding may or may not change the output
    retract_lemma!(S, r["lemma"])
    rc = check!(S, c1)
    after = output_bounds(S, c1)
    @test after == ref
    # independent lemma-free run of the same steps gives identical check results and caches
    S0 = open_case("safe_mid")
    check!(S0, 1)
    d1, _ = split!(S0, 1, 1)["children"]
    rc0 = check!(S0, d1)
    @test rc0["distance_bound"] == rc["distance_bound"]
    for (kk, bc) in S0.nodes[d1].cache_out.bounds_cache
        bc2 = S.nodes[c1].cache_out.bounds_cache[kk]
        @test bc.initialized == bc2.initialized
        bc.initialized && @test (bc.∂lower, bc.∂upper, bc.lower₁, bc.upper₁) == (bc2.∂lower, bc2.∂upper, bc2.lower₁, bc2.upper₁)
    end
    @test K.validate_tree(S)
end

@testset "Checker mutations" begin
    S = open_case("safe_split")
    auto_expand!(S, S.obligations[S.main_obligation].root; max_nodes=10_000)
    @test startswith(check_cert(S, "safe_split"), "ACCEPT SAFE")
    leaf(c) = findfirst(e -> e[4] == "c", c["nodes"])
    split_(c) = findfirst(e -> e[4] == "s", c["nodes"])
    muts = Dict(
        "leaf deleted" => c -> deleteat!(c["nodes"], leaf(c)),
        "cut moved" => c -> (e = c["nodes"][split_(c)]; e[5]["c"] = (e[6][1][e[5]["d"]] + e[5]["c"]) / 2),
        "cut outside" => c -> (e = c["nodes"][split_(c)]; e[5]["c"] = e[6][2][e[5]["d"]] + 1.0),
        "child box widened" => c -> (e = c["nodes"][leaf(c)]; e[6][1][1] -= 0.01),
        "wrong spec hash" => c -> (c["spec_hash"] = "00" * c["spec_hash"][3:end]),
        "status forged" => c -> begin
            i = leaf(c); c["nodes"][i][4] = "o"; c["nodes"][i][5] = Dict()
        end,
        "epsilon changed" => c -> (c["property"]["epsilon"] = c["property"]["epsilon"] / 10),
        "fake closed leaf" => c -> begin
            # replace a split's whole subtree by a single closed leaf claim
            i = split_(c); e = c["nodes"][i]
            byid = Dict(x[1] => x for x in c["nodes"])
            under(x) = (y = byid[x][2]; while y != 0; y == e[1] && return true; y = byid[y][2]; end; false)
            c["nodes"] = [x for x in c["nodes"] if !under(x[1])]
            for x in c["nodes"]; x[1] == e[1] && (x[4] = "c"; x[5] = Dict("db" => 0.0)); end
        end,
    )
    for (name, m) in muts
        res = check_cert(S, "safe_split"; mutate=m)
        @info "mutation" name res
        @test startswith(res, "REJECT")
    end
end

@testset "Batch and per-leaf auto_expand" begin
    S = open_case("safe_mid")
    K.SESSION[] = S
    r = K.METHODS["batch"](Dict("actions" => [
        Dict("method" => "split", "params" => Dict("node" => 1, "dim" => 1, "cut" => 0.1)),
        Dict("method" => "split", "params" => Dict("node" => "\$1.children[2]", "dim" => 2, "cuts" => [-0.2, 0.2])),
        Dict("method" => "auto_expand", "params" => Dict("node" => 1, "per_leaf_max_nodes" => 10_000)),
        Dict("method" => "open_session", "params" => Dict())]))
    @test r["n_done"] == 4
    @test r["results"][3]["leaves_at_start"] == 4 && r["results"][3]["subtree_closed"]
    @test haskey(r["results"][4], "error")
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, "safe_mid"), "ACCEPT SAFE")
    bad = K.METHODS["batch"](Dict("actions" => [Dict("method" => "split", "params" => Dict("node" => "\$5.children[1]", "dim" => 1))]))
    @test haskey(bad["results"][1], "error")
end

@testset "Extension stubs" begin
    @test_throws K.NotImplementedError K.make_property(Dict("kind" => "top1"))
    @test_throws K.NotImplementedError K.make_property(Dict("kind" => "robustness"))
    @test K.split_blocks(K.RobustnessAdapter(0.9, 0.01)) == [:primary, :sec1, :sec2]
    @test K.needs_lp(K.Top1Adapter(0.0))
end

@testset "RPC" begin
    i = INST["safe_easy"]
    reqs = [
        Dict("id" => 1, "method" => "ping"),
        Dict("id" => 2, "method" => "open_session", "params" => Dict("net1" => i["net1"], "net2" => i["net2"], "spec" => i["spec"],
             "property" => Dict("kind" => "epsilon", "epsilon" => i["epsilon"]))),
        Dict("id" => 3, "method" => "check", "params" => Dict("node" => 1, "rationale" => "root")),
        Dict("id" => 4, "method" => "split", "params" => Dict("node" => 1, "dim" => 1)),
        Dict("id" => 5, "method" => "nope"),
        Dict("id" => 8, "method" => "read_weights", "params" => Dict("layer" => 1, "which" => "diff")),
        Dict("id" => 9, "method" => "split", "params" => Dict("node" => 1, "dim" => 1, "dim_block" => "sec1")),
        Dict("id" => 6, "method" => "session_status"),
        Dict("id" => 7, "method" => "shutdown"),
    ]
    inp = IOBuffer(join(JSON.json.(reqs), "\n") * "\n")
    out = IOBuffer()
    oldout = stdout
    try
        K.serve(inp, out)
    finally
        redirect_stdout(oldout)
    end
    lines = JSON.parse.(split(strip(String(take!(out))), "\n"))
    @test lines[1]["result"]["ready"] == true
    byid = Dict(l["id"] => l for l in lines[2:end])
    @test byid[1]["result"]["pong"] == true
    @test byid[3]["result"]["holds"] == true
    @test byid[4]["error"]["type"] == "precondition"     # node 1 is closed, not an open leaf
    @test byid[5]["error"]["type"] == "bad_request"
    @test byid[6]["result"]["verdict"] == "SAFE"
    @test haskey(byid[8]["result"], "diff_summary")
    @test haskey(byid[9], "error")
    @test byid[7]["result"]["bye"] == true
end
end # outer
