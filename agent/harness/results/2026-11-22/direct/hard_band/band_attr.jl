using VeryDiffAgentKernel, JSON; const K=VeryDiffAgentKernel
d="/tmp/claude-0/mnist"
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=split_relu!(S,1,3,45;var="z1",t=0.95)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=1.05)["children"][1]
x=check!(S,n); println("band bound ", x["distance_bound"])
a=difference_attribution(S,n,7,4;n_samples=60000,top_k=12)
println("kept ",a["samples_kept"]," max diff ",a["max_difference"]," independent sum ",a["independent_max_sum"])
for u in a["top_loss_units"]; println(JSON.json(u)); end
