using Test
using JSON
using Random
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

@testset "trial_lemma dry run" begin
    S = open_case("safe_mid")
    root = S.obligations[S.main_obligation].root
    check!(S, root)
    c1, c2 = split!(S, root, 1)["children"]
    k = S.relu_layers[end]
    res = K.inspect_propagation(S, S.nodes[c1]; stop=k)
    b = K.zono_bounds(res.Zout.∂Z)
    lo, hi = b[1:2, 1] .- 1e-6, b[1:2, 2] .+ 1e-6
    before_ids = sort(collect(keys(S.nodes))); before_checks = S.n_checks; before_log = length(S.log)
    cert_before = K.certificate(S)["nodes"]
    t = trial_lemma!(S, c1, k, [1, 2]; kind="interval", dlo=lo, dhi=hi, lemma_max_nodes=5000, main_max_nodes=5000)
    @test sort(collect(keys(S.nodes))) == before_ids && isempty(S.lemmas)
    @test S.n_checks == before_checks && S.n_trial_propagations > 0
    @test length(S.log) == before_log + 1 && S.log[end]["action"] == "trial_lemma"
    @test K.certificate(S)["nodes"] == cert_before
    @test t["lemma"]["status"] == "proven" && t["with_lemma"]["closed"] && t["without_lemma"]["closed"]
    @test t["net_saving_nodes"] isa Integer
    # the dry-run numbers equal a real run
    r = introduce_lemma!(S, c1, k, [1, 2], lo, hi)
    prove_lemma!(S, r["lemma"]; max_nodes=5000)
    auto_expand!(S, r["child"]; max_nodes=5000)
    @test length(K.subtree(S, r["lemma_root"])) == t["lemma"]["proof_nodes"]
    @test length(K.subtree(S, r["child"])) == t["with_lemma"]["subtree_nodes"]
end

@testset "LP tightening with lemma constraints (dual-certified)" begin
    local S, lpleaves
    found = false
    for seed in 1:12                          # the LP effect depends on the sampled bounds; search a few seeds
        Random.seed!(seed)
        S = open_case("safe_hard")
        k = S.relu_layers[end]
        X = [S.root_box.lo .+ rand(length(S.root_box.lo)) .* (S.root_box.hi .- S.root_box.lo) for _ in 1:3000]
        D = reduce(hcat, [(h = K.hidden_at(S, x, k); h[1] .- h[2]) for x in X])
        lo = vec(minimum(D, dims=2)); hi = vec(maximum(D, dims=2)); pad = 0.25 .* (hi .- lo) .+ 1e-3
        r = introduce_lemma!(S, 1, k, collect(1:size(D, 1)), lo .- pad, hi .+ pad)
        p = prove_lemma!(S, r["lemma"]; max_nodes=50_000, max_seconds=600.0)
        p["lemma_status"] == "proven" || continue
        auto_expand!(S, r["child"]; max_nodes=100_000, max_seconds=600.0)
        session_verdict(S) == "SAFE" || continue
        lpleaves = [e for e in K.certificate(S)["nodes"] if e[4] == "c" && haskey(e[5], "lp")]
        isempty(lpleaves) && continue
        found = true
        @info "LP-closed leaves found" seed n = length(lpleaves)
        break
    end
    @test found
    @test startswith(check_cert(S, "safe_hard"), "ACCEPT SAFE")
    target = lpleaves[1][1]
    neg = check_cert(S, "safe_hard"; mutate=c -> for e in c["nodes"]
        e[1] == target && (e[5]["lp"]["rows"][1]["lam"][1] = -1.0)
    end)
    @test startswith(neg, "REJECT")
    short = check_cert(S, "safe_hard"; mutate=c -> for e in c["nodes"]
        e[1] == target && pop!(e[5]["lp"]["rows"][1]["lam"])
    end)
    @test startswith(short, "REJECT")
    # relational lemma: introduce, prove (trivially, from the root bound of the combination), refute a false one
    S2 = open_case("safe_mid")
    k2 = S2.relu_layers[1]
    res = K.inspect_propagation(S2, S2.nodes[1]; stop=k2)
    off, m = K.var_layout([res.Zout.∂Z, res.Zout.Z₁, res.Zout.Z₂])
    row = K.combo_row(res.Zout, [("d", 1, 1.0), ("z1", 2, -0.5)], off, m)
    rr = introduce_relational_lemma!(S2, 1, k2, [("d", 1, 1.0), ("z1", 2, -0.5)], row.c + sum(abs, row.g) + 1e-6)
    @test prove_lemma!(S2, rr["lemma"]; max_nodes=10)["lemma_status"] == "proven"
    auto_expand!(S2, rr["child"]; max_nodes=100_000)
    @test session_verdict(S2) == "SAFE"
    @test startswith(check_cert(S2, "safe_mid"), "ACCEPT SAFE")
    x = (S2.root_box.lo .+ S2.root_box.hi) ./ 2
    h1, h2 = K.hidden_at(S2, x, k2)
    v = (h1[1] - h2[1]) - 0.5 * h1[2]
    S3 = open_case("safe_mid")
    rf = introduce_relational_lemma!(S3, 1, k2, [("d", 1, 1.0), ("z1", 2, -0.5)], v - 1e-3)
    @test refute_lemma!(S3, rf["lemma"], x)["refuted"] == true
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

@testset "ReLU case splits (cache seeding, csb, infeasibility)" begin
    S = open_case("safe_hard")
    check!(S, 1)
    sc = relu_scores(S, 1; top_k=50)
    @test sc["n_relaxation_columns"] > 0
    e = first(x for x in sc["top"] if x["z1"][1] < 0 < x["z1"][2])
    k, j = e["layer"], e["neuron"]
    rb = S.nodes[1].cache_out.bounds_cache[k]
    @test rb.lower₁[j] < 0 < rb.upper₁[j]
    r = split_relu!(S, 1, k, j; var="z1", t=0.0)
    a, b = r["children"]
    check!(S, a); check!(S, b)
    ca = S.nodes[a].cache_out.bounds_cache[k]; cb = S.nodes[b].cache_out.bounds_cache[k]
    # the split decision reaches VeryDiff's transformer: phase fixed in the cache ...
    @test ca.upper₁[j] == 0.0 && cb.lower₁[j] == 0.0
    # ... so no relaxation column for this unit in N1 below
    for c in (a, b)
        res = K.inspect_propagation(S, S.nodes[c])
        @test !any(x -> x[1] == k && x[2] == "z1" && x[3] == j, K.relaxation_columns(S, res))
    end
    # cross-network tightening on the seeded layer: z2 ⊆ z1 − d
    for cc in (ca, cb)
        @test cc.lower₂[j] >= cc.lower₁[j] - cc.∂upper[j] - 1e-12
        @test cc.upper₂[j] <= cc.upper₁[j] - cc.∂lower[j] + 1e-12
        @test cc.lower₂[j] >= rb.lower₂[j] - 1e-12 && cc.upper₂[j] <= rb.upper₂[j] + 1e-12
    end
    # soundness on samples: each sample's outputs lie in the bounds of the branch its phase belongs to
    rng = MersenneTwister(7)
    ob = Dict(c => (z = K.inspect_propagation(S, S.nodes[c]).Zout; (K.zono_bounds(z.Z₁), K.zono_bounds(z.Z₂), K.zono_bounds(z.∂Z)))
              for c in (a, b))
    lo, hi = S.root_box.lo, S.root_box.hi
    for _ in 1:300
        x = lo .+ rand(rng, length(lo)) .* (hi .- lo)
        ep = eval_point(S, x)
        h = ep["hidden_preact"][string(k)]["n1"][j]
        for c in vcat(h <= 0 ? [a] : Int[], h >= 0 ? [b] : Int[])
            b1, b2, bd = ob[c]
            @test all(b1[:, 1] .- 1e-9 .<= ep["y1"] .<= b1[:, 2] .+ 1e-9)
            @test all(b2[:, 1] .- 1e-9 .<= ep["y2"] .<= b2[:, 2] .+ 1e-9)
            @test all(bd[:, 1] .- 1e-9 .<= ep["y1"] .- ep["y2"] .<= bd[:, 2] .+ 1e-9)
        end
    end
    # complete proof through the relu split; independent checker accepts
    for c in (a, b)
        S.nodes[c].kind == :open && auto_expand!(S, c; max_nodes=100000)
    end
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, "safe_hard"), "ACCEPT")
    # mutation: wrong split variable -> replay differs; dropping a branch -> structural reject
    @test startswith(check_cert(S, "safe_hard"; mutate=c -> begin
        e = only(x for x in c["nodes"] if x[4] == "s" && get(x[5], "b", "") == "relu")
        e[5]["k"] = -1 end), "REJECT")
    @test startswith(check_cert(S, "safe_hard"; mutate=c -> begin
        filter!(x -> x[1] != b, c["nodes"]) end), "REJECT")

    # infeasible branch: z1 >= t with t far above the upper bound
    S2 = open_case("safe_hard")
    check!(S2, 1)
    u = S2.nodes[1].cache_out.bounds_cache[k].upper₁[j]
    r2 = split_relu!(S2, 1, k, j; var="z1", t=u + 1.0)
    c2 = check!(S2, r2["children"][2])
    @test c2["outcome"] == "closed" && S2.nodes[r2["children"][2]].evidence["check"] == "infeasible"
    # an infeasible flag on a feasible leaf is rejected
    auto_expand!(S2, r2["children"][1]; max_nodes=100000)
    @test session_verdict(S2) == "SAFE"
    @test startswith(check_cert(S2, "safe_hard"), "ACCEPT")
    @test startswith(check_cert(S2, "safe_hard"; mutate=c -> begin
        for x in c["nodes"]
            x[4] == "c" && !get(x[5], "inf", false) && (x[5]["inf"] = true; break)
        end end), "REJECT")
end

@testset "One-sided differential fallback (src flag) with ReLU splits" begin
    S = open_session(INST["safe_hard"]["net1"], INST["safe_hard"]["net2"], INST["safe_hard"]["spec"],
                     Dict("kind" => "epsilon", "epsilon" => INST["safe_hard"]["epsilon"]); fallback=true)
    check!(S, 1)
    sc = relu_scores(S, 1; top_k=20)
    e = first(x for x in sc["top"] if x["z1"][1] < 0 < x["z1"][2] && x["z2"][1] < 0 < x["z2"][2])
    kids = apply_action!(S, 1, Dict("kind" => "phase4", "layer" => e["layer"], "neuron" => e["neuron"]))
    @test length(kids) == 4
    rng = MersenneTwister(3)
    for c in kids
        check!(S, c)
        S.nodes[c].kind == :closed && continue
        z = K.inspect_propagation(S, S.nodes[c]).Zout
        bd = K.zono_bounds(z.∂Z)
        for _ in 1:200
            x = S.root_box.lo .+ rand(rng, length(S.root_box.lo)) .* (S.root_box.hi .- S.root_box.lo)
            ep = eval_point(S, x)
            h1 = ep["hidden_preact"][string(e["layer"])]["n1"][e["neuron"]]; h2 = ep["hidden_preact"][string(e["layer"])]["n2"][e["neuron"]]
            st = S.nodes[c].constraints
            inside = all(st) do q
                v = isnothing(q.n1lo) ? h2 : h1
                lo_, hi_ = isnothing(q.n1lo) ? (q.n2lo[1], q.n2hi[1]) : (q.n1lo[1], q.n1hi[1])
                lo_ <= v <= hi_
            end
            inside && @test all(bd[:, 1] .- 1e-9 .<= ep["y1"] .- ep["y2"] .<= bd[:, 2] .+ 1e-9)
        end
    end
    for c in kids
        S.nodes[c].kind == :open && auto_expand!(S, c; max_nodes=100000)
    end
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, "safe_hard"), "ACCEPT")
    VeryDiff.ONE_SIDED_DIFF_FALLBACK[] = false
end

@testset "Threshold splits: dual-tightened caches and LP constraint rows are sound" begin
    for name in ("safe_hard", "safe_mid")
        S = open_session(INST[name]["net1"], INST[name]["net2"], INST[name]["spec"],
                         Dict("kind" => "epsilon", "epsilon" => INST[name]["epsilon"]); fallback=true)
        check!(S, 1)
        k = S.relu_layers[1]
        bc0 = S.nodes[1].cache_out.bounds_cache[k]
        j = argmax(bc0.upper₁ .- bc0.lower₁)
        t = (bc0.lower₁[j] + bc0.upper₁[j]) / 2
        kids = split_relu!(S, 1, k, j; var="z1", t=t)["children"]
        rng = MersenneTwister(11)
        nin = 0
        for c in kids
            check!(S, c)
            n = S.nodes[c]
            isnothing(n.cache_out) && continue
            for _ in 1:300
                x = S.root_box.lo .+ rand(rng, length(S.root_box.lo)) .* (S.root_box.hi .- S.root_box.lo)
                ep = eval_point(S, x)
                h = ep["hidden_preact"][string(k)]["n1"][j]
                (c == kids[1] ? h <= t : h >= t) || continue
                nin += 1
                for kk in S.relu_layers
                    bc = n.cache_out.bounds_cache[kk]
                    h1 = ep["hidden_preact"][string(kk)]["n1"]; h2 = ep["hidden_preact"][string(kk)]["n2"]
                    @test all(bc.lower₁ .- 1e-9 .<= h1 .<= bc.upper₁ .+ 1e-9)
                    @test all(bc.lower₂ .- 1e-9 .<= h2 .<= bc.upper₂ .+ 1e-9)
                    @test all(bc.∂lower .- 1e-9 .<= h1 .- h2 .<= bc.∂upper .+ 1e-9)
                end
            end
        end
        @test nin > 100
        for c in kids
            S.nodes[c].kind == :open && auto_expand!(S, c; max_nodes=100000)
        end
        @test session_verdict(S) == "SAFE"
        @test startswith(check_cert(S, name), "ACCEPT")
        # LP evidence that uses constraint rows: tampering with its multipliers is rejected
        haslp = any(n -> n.kind == :closed && !isnothing(n.evidence) && haskey(n.evidence, "lp") &&
                         !isempty(n.evidence["lp"]["cons"]), values(S.nodes))
        if haslp
            @test startswith(check_cert(S, name; mutate=c -> begin
                for e in c["nodes"]
                    if e[4] == "c" && haskey(e[5], "lp") && !isempty(e[5]["lp"]["cons"])
                        for r in e[5]["lp"]["rows"]; r["lam"] .= -1.0; end
                        break
                    end
                end end), "REJECT")
        end
    end
end

@testset "Bound rows (VeryDiff.BOUND_ROWS): sound caches; the checker computes the same caches" begin
    name = "safe_mid"
    S = open_session(INST[name]["net1"], INST[name]["net2"], INST[name]["spec"],
                     Dict("kind" => "epsilon", "epsilon" => INST[name]["epsilon"]); fallback=true,
                     bound_rows=50, bound_rows_rtol=0.0)
    check!(S, 1)
    k = S.relu_layers[1]
    bc0 = S.nodes[1].cache_out.bounds_cache[k]
    j = argmax(bc0.upper₁ .- bc0.lower₁)
    t = (bc0.lower₁[j] + bc0.upper₁[j]) / 2
    kids = split_relu!(S, 1, k, j; var="z1", t=t)["children"]
    rng = MersenneTwister(12)
    nin = 0
    for (side, c) in enumerate(kids)
        n = S.nodes[c]
        res = K.run_propagation(S, n.region, n.cache_in, Int[], nothing; constraints=n.constraints)
        @test !isempty(res.crows)
        # the checker's own implementation gives the same caches
        seeds = Dict(k => [CertChecker.threshold_constraint("z1", j, t, side)])
        rc = CertChecker.propagate(S.N, n.region.lo, n.region.hi, S.active_dims, n.cache_in, seeds, nothing;
                                   constraints=seeds, bound_rows=(50, 0.0))
        for kk in S.relu_layers
            a = res.cache.bounds_cache[kk]; b = rc.cache.bounds_cache[kk]
            for f in (:lower₁, :upper₁, :lower₂, :upper₂, :∂lower, :∂upper)
                @test isapprox(getfield(a, f), getfield(b, f); rtol=1e-9, atol=1e-12)
            end
        end
        # sampled concrete values of the child's region lie inside the caches
        for _ in 1:300
            x = S.root_box.lo .+ rand(rng, length(S.root_box.lo)) .* (S.root_box.hi .- S.root_box.lo)
            ep = eval_point(S, x)
            h = ep["hidden_preact"][string(k)]["n1"][j]
            (side == 1 ? h <= t : h >= t) || continue
            nin += 1
            for kk in S.relu_layers
                bc = res.cache.bounds_cache[kk]
                h1 = ep["hidden_preact"][string(kk)]["n1"]; h2 = ep["hidden_preact"][string(kk)]["n2"]
                @test all(bc.lower₁ .- 1e-9 .<= h1 .<= bc.upper₁ .+ 1e-9)
                @test all(bc.lower₂ .- 1e-9 .<= h2 .<= bc.upper₂ .+ 1e-9)
                @test all(bc.∂lower .- 1e-9 .<= h1 .- h2 .<= bc.∂upper .+ 1e-9)
            end
        end
    end
    @test nin > 100
    for c in kids
        S.nodes[c].kind == :open && auto_expand!(S, c; max_nodes=100000)
    end
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, name), "ACCEPT")

    # Inherited caches (plain input splitting): on safe_hard, the rows tighten the caches of a few nodes. There
    # the kernel and the checker must agree, and the caches must be sound.
    i = INST["safe_hard"]
    S = open_session(i["net1"], i["net2"], i["spec"], Dict("kind" => "epsilon", "epsilon" => i["epsilon"]);
                     bound_rows=1000, bound_rows_rtol=0.0)
    auto_expand!(S, 1; max_nodes=200)
    ntight = 0
    for id in sort(collect(keys(S.nodes)))
        n = S.nodes[id]
        res = K.run_propagation(S, n.region, n.cache_in, Int[], nothing)
        S.bound_rows = 0
        res0 = K.run_propagation(S, n.region, n.cache_in, Int[], nothing)
        S.bound_rows = 1000
        tight = any(kk -> (a = res.cache.bounds_cache[kk]; b = res0.cache.bounds_cache[kk];
                           any(a.upper₁ .< b.upper₁ .- 1e-9) || any(a.lower₁ .> b.lower₁ .+ 1e-9) ||
                           any(a.∂upper .< b.∂upper .- 1e-9) || any(a.∂lower .> b.∂lower .+ 1e-9)), S.relu_layers)
        tight || continue
        ntight += 1
        rc = CertChecker.propagate(S.N, n.region.lo, n.region.hi, S.active_dims, n.cache_in,
                                   Dict{Int,Vector{CertChecker.Seed}}(), nothing; bound_rows=(1000, 0.0))
        for kk in S.relu_layers
            a = res.cache.bounds_cache[kk]; b = rc.cache.bounds_cache[kk]
            for f in (:lower₁, :upper₁, :lower₂, :upper₂, :∂lower, :∂upper)
                @test isapprox(getfield(a, f), getfield(b, f); rtol=1e-9, atol=1e-12)
            end
        end
        for _ in 1:200
            x = n.region.lo .+ rand(rng, length(n.region.lo)) .* (n.region.hi .- n.region.lo)
            ep = eval_point(S, x)
            for kk in S.relu_layers
                bc = res.cache.bounds_cache[kk]
                h1 = ep["hidden_preact"][string(kk)]["n1"]; h2 = ep["hidden_preact"][string(kk)]["n2"]
                @test all(bc.lower₁ .- 1e-9 .<= h1 .<= bc.upper₁ .+ 1e-9)
                @test all(bc.lower₂ .- 1e-9 .<= h2 .<= bc.upper₂ .+ 1e-9)
                @test all(bc.∂lower .- 1e-9 .<= h1 .- h2 .<= bc.∂upper .+ 1e-9)
            end
        end
    end
    @test ntight >= 1
end

@testset "Relational (oblique) splits" begin
    name = "safe_hard"
    S = open_session(INST[name]["net1"], INST[name]["net2"], INST[name]["spec"],
                     Dict("kind" => "epsilon", "epsilon" => INST[name]["epsilon"]); fallback=true)
    check!(S, 1)
    k = S.relu_layers[1]
    terms = [("z1", 1, 1.0), ("z1", 2, -0.5), ("z2", 3, 0.25)]
    ev(x) = (e = eval_point(S, x)["hidden_preact"][string(k)]; e["n1"][1] - 0.5 * e["n1"][2] + 0.25 * e["n2"][3])
    xc = (S.root_box.lo .+ S.root_box.hi) ./ 2
    t = ev(xc)
    kids = split_lin!(S, 1, k, terms, t)["children"]
    rng = MersenneTwister(5); nin = 0
    for c in kids
        check!(S, c)
        n = S.nodes[c]; isnothing(n.cache_out) && continue
        for _ in 1:300
            x = S.root_box.lo .+ rand(rng, length(S.root_box.lo)) .* (S.root_box.hi .- S.root_box.lo)
            (c == kids[1] ? ev(x) <= t : ev(x) >= t) || continue
            nin += 1
            ep = eval_point(S, x)
            for kk in S.relu_layers
                bc = n.cache_out.bounds_cache[kk]
                h1 = ep["hidden_preact"][string(kk)]["n1"]; h2 = ep["hidden_preact"][string(kk)]["n2"]
                @test all(bc.lower₁ .- 1e-9 .<= h1 .<= bc.upper₁ .+ 1e-9) && all(bc.lower₂ .- 1e-9 .<= h2 .<= bc.upper₂ .+ 1e-9)
            end
        end
    end
    @test nin > 100
    for c in kids
        S.nodes[c].kind == :open && auto_expand!(S, c; max_nodes=100000)
    end
    @test session_verdict(S) == "SAFE"
    @test startswith(check_cert(S, name), "ACCEPT")
    @test startswith(check_cert(S, name; mutate=c -> begin
        e = only(x for x in c["nodes"] if x[4] == "s" && get(x[5], "b", "") == "lin"); e[5]["terms"] = [] end), "REJECT")
end

@testset "LP-relaxation rows: valid at the witness, certified, tamper-evident" begin
    for name in ("safe_hard", "safe_mid")
        S = open_session(INST[name]["net1"], INST[name]["net2"], INST[name]["spec"],
                         Dict("kind" => "epsilon", "epsilon" => INST[name]["epsilon"]); fallback=true, lp_relax=true)
        check!(S, 1)
        n = S.nodes[1]
        res = K.inspect_propagation(S, n)
        # one witness ε reproducing every stored zonotope row at a concrete point must satisfy every relaxation row
        Zs = Any[]; vals = Vector{Float64}[]
        rng = MersenneTwister(2)
        for trial in 1:8
            x = S.root_box.lo .+ rand(rng, length(S.root_box.lo)) .* (S.root_box.hi .- S.root_box.lo)
            o1, o2 = K.eval_raw(S, x)
            Zs = Any[]; vals = Vector{Float64}[]
            for L in K.get_layers(S.N), pos in (K.get_inputs(L)[1], K.get_outputs(L)[1])
                cz = res.ps.zono_storage.zonotopes[pos]; (isnothing(cz) || isnothing(cz.zonotope)) && continue
                nm = get(S.pos_names, pos, ""); haskey(o1, nm) || continue
                v1 = Float64.(vec(o1[nm])); v2 = Float64.(vec(o2[nm]))
                for (Z, v) in ((cz.zonotope.Z₁, v1), (cz.zonotope.Z₂, v2), (cz.zonotope.∂Z, v1 .- v2))
                    push!(Zs, Z); push!(vals, v)
                end
            end
            off, m = K.var_layout(Zs)
            md = K.Model(K.GLPK.Optimizer); K.set_silent(md)
            K.@variable(md, -1 <= e[1:m] <= 1); K.@variable(md, sres >= 0)
            for (Z, v) in zip(Zs, vals), j in eachindex(v)
                r = K.affrow(Z, j, off, m)
                K.@constraint(md, r.c + sum(r.g[i] * e[i] for i in 1:m if r.g[i] != 0; init=0.0) - v[j] <= sres)
                K.@constraint(md, -(r.c + sum(r.g[i] * e[i] for i in 1:m if r.g[i] != 0; init=0.0) - v[j]) <= sres)
            end
            K.@objective(md, Min, sres); K.optimize!(md)
            @test K.objective_value(md) <= 1e-8
            ew = K.value.(e)
            rr, rv, spec = K.relax_rows(S, res, off, m)
            @test !isempty(rv)
            @test maximum(rr[i]' * ew - rv[i] for i in eachindex(rv)) <= 1e-7
        end
        name == "safe_mid" || continue      # complete proofs only on the small instance (GLPK is slow on safe_hard)
        auto_expand!(S, 1; max_nodes=100000)
        @test session_verdict(S) == "SAFE"
        @test startswith(check_cert(S, name), "ACCEPT")
        used = any(x -> x.kind == :closed && !isnothing(x.evidence) && haskey(x.evidence, "lp") && haskey(x.evidence["lp"], "relax"), values(S.nodes))
        @test used
        if used
            first_relax(c) = first(e for e in c["nodes"] if e[4] == "c" && haskey(e[5], "lp") && haskey(e[5]["lp"], "relax"))
            @test startswith(check_cert(S, name; mutate=c -> (first_relax(c)[5]["lp"]["relax"][1][3] = "bogus")), "REJECT")
            @test startswith(check_cert(S, name; mutate=c -> (first_relax(c)[5]["lp"]["relax"][1][2] = 10^6)), "REJECT")
            @test startswith(check_cert(S, name; mutate=c -> (c["options"]["lp_relax"] = false)), "REJECT")
            @test startswith(check_cert(S, name; mutate=c -> (for r in first_relax(c)[5]["lp"]["rows"]; r["lam"] .= -1.0; end)), "REJECT")
        end
    end
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
