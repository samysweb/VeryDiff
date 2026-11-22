using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true, lp_relax=true)
check!(S,1)
for (a,b) in ((0.058,1.133),(0.975,1.0),(0.975,0.9875),(0.975,0.98125),(0.98,0.981))
  s=K.local_snapshot(S,1)
  n=1; a>0.06 && (n=split_relu!(S,n,3,45;var="z1",t=a)["children"][2]); b<1.13 && (n=split_relu!(S,n,3,45;var="z1",t=b)["children"][1])
  t=time(); x=check!(S,n); dt=time()-t
  nd=S.nodes[n]; nrows = nd.kind==:closed && haskey(nd.evidence,"lp") ? length(nd.evidence["lp"]["relax"]) : -1
  bnd = nd.kind==:closed ? x["distance_bound"] : (res=K.inspect_propagation(S,nd); 1+K.lp_fallback(S,nd,S.obligations[1],res)[3])
  println("r∈[$a,$b]: $(x["outcome"]) bound $(round(bnd,digits=4)) check $(round(dt,digits=2)) s  relax rows $nrows"); flush(stdout)
  K.local_restore!(S,1,s)
end
