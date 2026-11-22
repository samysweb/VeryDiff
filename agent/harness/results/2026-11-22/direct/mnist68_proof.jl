# Proof script: MNIST mnist_3_100 vs mnist_3_100_pruned5, mnist_68_local_15, ε = 1.0 (max_i |N1(x)_i − N2(x)_i| ≤ 1).
#
# Reading the program (weights; see network_diff_structure, read_weights, difference_attribution):
#  (F1) N2 = N1 with layer-1 units {31,38,45,80,95} and layer-2 units {10,37,38,51,86} knocked out (zero weights,
#       bias −1, so they are off in N2 everywhere); all other weights agree to 4.2e-4.
#  (F2) Only 15 pixels are free (each in [0,1]); layer-1 pre-activations are affine in them. Exactly over the box,
#       N1's pruned units 31,38,80,95 are always off, unit 45 is always on: r := h45(x) ∈ [0.058, 1.133].
#       So the only real layer-1 difference is r, and the layer-2 pre-activation difference is W[:,45]·r (+ noise).
#  (F3) The last layer is the identity on ReLU(h3); the critical output is 4 (h3_4 ∈ [10.8, 13.5], always on), so
#       diff_4 is linear in the layer-2 activations. Sampled: diff_4 ≈ 0.2 + 0.65 r (corr 0.97), max 0.923 at
#       r ≈ 1.13. 21 layer-2 units have mixed phases; adding their terms' maxima independently gives 1.10 —
#       this "independent maximisation" (plus relaxation) is why VeryDiff's root bound is 1.187.
# Case distinction: r is the one quantity that drives the difference, so split on r (= z1 of unit 45 at the
# input of ReLU layer 3) into bands. Inside a band, the constraint tightens every later bound (dual bounds with the
# band rows) and the final check (LP rows): bands with r ≤ 0.85 close with a single propagation each; the bands
# above (true max 0.87–0.92, margin < 0.13) are discharged by VeryDiff's input splitting under the band
# constraint. Band [0.95,1.05] was refined to 0.025-wide bands after it did not close within 30k checks.
using VeryDiffAgentKernel
d = get(ENV, "INST", "/tmp/claude-0/mnist")
S = open_session("$d/net1.onnx", "$d/net2.onnx", "$d/spec.vnnlib", Dict("kind" => "epsilon", "epsilon" => 1.0); fallback=true)
cuts = [0.55, 0.65, 0.75, 0.85, 0.95, 0.975, 1.0, 1.025, 1.05]
node = 1; bands = Int[]
for t in cuts
    r = split_relu!(S, node, 3, 45; var="z1", t=t, rationale="band on r = h45 (the only layer-1 difference), cut at $t")
    push!(bands, r["children"][1]); global node = r["children"][2]
end
push!(bands, node)
for b in bands
    check!(S, b)                                                   # low bands close here (1 propagation)
    S.nodes[b].kind == :open && auto_expand!(S, b; max_nodes=200_000, max_seconds=7200.0)   # VeryDiff discharges the rest
end
println("verdict ", session_verdict(S), ", propagations ", S.n_checks, ", nodes ", length(S.nodes))
println(export_certificate(S, get(ENV, "OUT", "mnist68_weight_reasoning_cert.json.gz")))
