using VeryDiffAgentKernel, JSON
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0))
r=network_diff_structure(S); for l in r["layers"]; println(JSON.json(l)); end
a=difference_attribution(S,1,7,4;n_samples=20000,top_k=5)
println(JSON.json(Dict(k=>v for (k,v) in a if k!="top_loss_units"))); for u in a["top_loss_units"]; println(JSON.json(u)); end
