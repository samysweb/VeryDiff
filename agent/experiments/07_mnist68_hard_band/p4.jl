using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=split_relu!(S,1,3,45;var="z1",t=0.975)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=1.0)["children"][1]
check!(S,n)
lpb(c)=(x=S.nodes[c]; x.kind==:closed ? (x.evidence["check"]=="infeasible" ? "inf" : round(x.distance_bound,digits=3)) : (res=K.inspect_propagation(S,x); round(1+K.lp_fallback(S,x,S.obligations[1],res)[3],digits=3)))
rs=relu_scores(S,n;top_k=12)
for e in rs["top"]
  e["var"]=="d" || continue
  s=K.local_snapshot(S,n); kids=apply_action!(S,n,Dict("kind"=>"phase4","layer"=>6,"neuron"=>e["neuron"]))
  bs=[(check!(S,c); lpb(c)) for c in kids]; K.local_restore!(S,n,s)
  println("unit $(e["neuron"]) d-mass $(e["score"]) z1 $(e["z1"]) z2 $(e["z2"]): (z1≤0,z2≤0) (z1≤0,z2≥0) (z1≥0,z2≤0) (z1≥0,z2≥0) = $bs"); flush(stdout)
end
