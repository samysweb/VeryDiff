# Builds docs/examples/example_certificate.json: a small certificate on the tiny test instance `safe_split`
# that uses every feature of the MNIST certificate (see example_certificate.jsonc for the explanation).
#   VERYDIFF_NO_GUROBI=1 julia --project=agent/julia/VeryDiffAgentKernel agent/docs/examples/make_example_certificate.jl
using VeryDiffAgentKernel
const NETS = joinpath(@__DIR__, "..", "..", "tests", "nets", "safe_split")
S = open_session(joinpath(NETS, "net1.onnx"), joinpath(NETS, "net2.onnx"), joinpath(NETS, "spec.vnnlib"),
                 Dict("kind" => "epsilon", "epsilon" => 0.064); fallback=true, lp_relax=true)
relu = S.relu_layers[1]
# 1. Case split on a hidden unit: z1 of unit 1 at the input of the first ReLU layer, threshold -0.33. Not checked.
a, b = split_relu!(S, 1, relu, 1; var="z1", t=-0.33, rationale="band on the first hidden unit")["children"]
# 2. Lower band: VeryDiff closes it directly.
check!(S, a)
# 3. Upper band: check it (it stays open), then split input dimension 1 at the midpoint.
check!(S, b)
b1, b2 = split!(S, b, 1; rationale="input split inside the upper band")["children"]
# 4. Both halves close only with LP evidence (split constraint + relaxation rows).
check!(S, b1); check!(S, b2)
println(session_status(S)["verdict"], " ", [(id, S.nodes[id].kind, get(something(S.nodes[id].evidence, Dict()), "check", "")) for id in sort(collect(keys(S.nodes)))])
println(export_certificate(S, joinpath(@__DIR__, "example_certificate.json")))
