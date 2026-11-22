using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
cuts=parse.(Float64,split(ARGS[1],",")); CAP=parse(Int,get(ENV,"CAP","30000"))
node=1; bands=Int[]
for t in cuts; r=split_relu!(S,node,3,45;var="z1",t=t); push!(bands,r["children"][1]); global node=r["children"][2]; end
push!(bands,node)
lo=0.058; total=0
for (i,b) in enumerate(bands)
  hi= i<=length(cuts) ? cuts[i] : 1.133
  c0=S.n_checks; t=time()
  a=auto_expand!(S,b;max_nodes=CAP,max_seconds=7200.0)
  used=S.n_checks-c0; global total+=used
  println("band [$lo,$hi]: $(is_closed(S,b) ? "closed" : "OPEN") with $used checks, $(round(time()-t,digits=1)) s"); flush(stdout)
  global lo=hi
end
println("total checks ", total, " verdict ", session_verdict(S), " nodes ", length(S.nodes))
session_verdict(S)=="SAFE" && println(export_certificate(S, get(ENV,"OUT","$(d)/bands_cert.json.gz")))
