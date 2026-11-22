using VeryDiffAgentKernel, JSON; const K=VeryDiffAgentKernel
d=ARGS[1]; eps=parse(Float64,ARGS[2])
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps))
auto_expand!(S,1;max_nodes=10^6)
# pick split nodes whose subtree has 10..60 nodes
sizes=Dict(id=>length(K.subtree(S,id)) for (id,n) in S.nodes if n.kind==:split)
picks=sort([id for (id,s) in sizes if 15<=s<=60])[1:4:end][1:min(6,end)]
for id in picks
  K.prune!(S,id); n=S.nodes[id]
  check!(S,id)
  unst=[(k,count((bc.lower₁ .< 0) .& (bc.upper₁ .> 0)),count((bc.lower₂ .< 0) .& (bc.upper₂ .> 0))) for (k,bc) in sort(collect(n.cache_out.bounds_cache);by=first) if k in S.relu_layers]
  rs=relu_scores(S,id;top_k=6); ss=split_scores(S,id;top_k=3)
  cands=Any[Dict("kind"=>"split","dim"=>e["dim"]) for e in ss["scores"]]
  for e in rs["top"]; e["z1"][1]<0<e["z1"][2] && e["z2"][1]<0<e["z2"][2] && push!(cands,Dict("kind"=>"phase4","layer"=>e["layer"],"neuron"=>e["neuron"]));
     e["d"][1]<0<e["d"][2] && push!(cands,Dict("kind"=>"split_relu","layer"=>e["layer"],"neuron"=>e["neuron"],"var"=>"d")); end
  T=trial_actions!(S,id,cands;baseline=false,max_nodes=500)
  println("node $id depth $(n.depth) bound $(round(n.distance_bound,digits=3)) unstable $unst")
  println("   top relu ", [(e["layer"],e["var"],e["neuron"],e["max_row_share"]) for e in rs["top"]])
  for r in T["results"]; println("   ", JSON.json(r["action"]), " => ", r["closed"] ? r["checks"] : "open>500"); end
end
