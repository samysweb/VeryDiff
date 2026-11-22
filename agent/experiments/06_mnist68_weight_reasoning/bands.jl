using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
cuts=parse.(Float64,split(ARGS[1],","))
node=1; bands=Int[]
for t in cuts
  r=split_relu!(S,node,3,45;var="z1",t=t); push!(bands,r["children"][1]); global node=r["children"][2]
end
push!(bands,node)
lo=0.058
for (i,b) in enumerate(bands)
  x=check!(S,b); n=S.nodes[b]
  hi= i<=length(cuts) ? cuts[i] : 1.133
  lpb = n.kind==:open ? (res=K.inspect_propagation(S,n); 1.0+K.lp_fallback(S,n,S.obligations[n.obligation],res)[3]) : x["distance_bound"]
  println("band r∈[$lo,$hi]: zono $(round(x["distance_bound"],digits=3)) LP $(round(lpb,digits=3)) $(x["outcome"])")
  global lo=hi
end
