using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
check!(S,1); println("root ", S.nodes[1].distance_bound)
r=split_relu!(S,1,6,69;var="z2")
for c in r["children"]
  x=check!(S,c); n=S.nodes[c]
  println("child $c ", K.constraint_summary(n.constraints[end]), ": ", x["outcome"], " db=", x["distance_bound"], " ", isnothing(n.evidence) ? "" : n.evidence["check"])
  # LP bound even if not closed
  if n.kind==:open
    res=K.inspect_propagation(S,n); ok,ev,worst=K.lp_fallback(S,n,S.obligations[n.obligation],res)
    println("   LP: ok=$ok worst excess=$worst  → bound ", 1.0+worst)
  end
end
