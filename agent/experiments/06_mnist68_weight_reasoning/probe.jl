using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
"band node for r ∈ [a,b]"
function band(a,b)
  n=1
  if a>0.06; n=split_relu!(S,n,3,45;var="z1",t=a)["children"][2]; end
  if b<1.13; n=split_relu!(S,n,3,45;var="z1",t=b)["children"][1]; end
  n
end
lpb(n)=(x=S.nodes[n]; x.kind==:closed ? round(x.distance_bound,digits=3) : (res=K.inspect_propagation(S,x); round(1.0+K.lp_fallback(S,x,S.obligations[x.obligation],res)[3],digits=3)))
for (a,b) in ((0.85,0.9),(0.9,0.95),(0.95,1.0),(1.0,1.05),(1.05,1.09),(1.09,1.133))
  s0=K.local_snapshot(S,1)
  n=band(a,b); check!(S,n); line="r∈[$a,$b]: $(lpb(n))"
  # phase splits on N2 units suggested by the loss analysis (1-based neuron = j+1)
  for j in (68,33,49,75)
    s=K.local_snapshot(S,n); kids=split_relu!(S,n,6,j+1;var="z2")["children"]
    bs=[(check!(S,c); lpb(c)) for c in kids]; K.local_restore!(S,n,s)
    line*="  z2[$j]≤0/≥0: $(bs)"
  end
  println(line); flush(stdout)
  K.local_restore!(S,1,s0)
end
