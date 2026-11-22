using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
check!(S,1)
for t in (0.5,0.7,0.8,0.9,1.0)
  s=K.local_snapshot(S,1)
  r=split_relu!(S,1,3,45;var="z1",t=t)
  out=String[]
  for c in r["children"]
    x=check!(S,c); n=S.nodes[c]; b=x["distance_bound"]
    lpb = n.kind==:open ? (res=K.inspect_propagation(S,n); 1.0+K.lp_fallback(S,n,S.obligations[n.obligation],res)[3]) : b
    push!(out, "$(c==r["children"][1] ? "r<=" : "r>=")$t: zono $(round(b,digits=3)) LP $(round(lpb,digits=3)) $(x["outcome"])")
  end
  println(join(out,"   |   ")); K.local_restore!(S,1,s)
end
