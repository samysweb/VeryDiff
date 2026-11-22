using VeryDiffAgentKernel, JSON; const K=VeryDiffAgentKernel
d=ARGS[1]; eps=parse(Float64,ARGS[2])
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps))
t=time(); check!(S,1); println("root bound ", S.nodes[1].distance_bound, " t=", time()-t)
n=S.nodes[1]
println("unstable ", [(k,count((bc.lower₁ .< 0) .& (bc.upper₁ .> 0)),count((bc.lower₂ .< 0) .& (bc.upper₂ .> 0))) for (k,bc) in sort(collect(n.cache_out.bounds_cache);by=first) if k in S.relu_layers])
rs=relu_scores(S,1;top_k=12); println("crit ", rs["critical_rows"], " ", rs["row_bounds"], " mass ", rs["row_generator_mass"])
for e in rs["top"]; println("  ", e["layer"]," ",e["var"]," ",e["neuron"]," share ",e["max_row_share"]," z1 ",e["z1"]," z2 ",e["z2"]," d ",e["d"]); end
ss=split_scores(S,1;top_k=3)
cands=Any[Dict("kind"=>"split","dim"=>e["dim"]) for e in ss["scores"]]
for e in rs["top"][1:8]
  x1=e["z1"][1]<0<e["z1"][2]; x2=e["z2"][1]<0<e["z2"][2]
  x1 && push!(cands,Dict("kind"=>"split_relu","layer"=>e["layer"],"neuron"=>e["neuron"],"var"=>"z1"))
  x2 && push!(cands,Dict("kind"=>"split_relu","layer"=>e["layer"],"neuron"=>e["neuron"],"var"=>"z2"))
  x1 && x2 && push!(cands,Dict("kind"=>"phase4","layer"=>e["layer"],"neuron"=>e["neuron"]))
end
# cheap proxy first: root bound after each action (children checked once)
for a in cands
  snap=K.local_snapshot(S,1); kids=apply_action!(S,1,a); bs=[ (check!(S,c); S.nodes[c].distance_bound) for c in kids]
  K.local_restore!(S,1,snap); println("  ", JSON.json(a), " child bounds ", round.(bs,digits=3))
end
