using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true, lp_relax=true)
cuts=isempty(ARGS[1]) ? Float64[] : parse.(Float64,split(ARGS[1],","))
node=1; bands=Int[]
for t in cuts; r=split_relu!(S,node,3,45;var="z1",t=t); push!(bands,r["children"][1]); global node=r["children"][2]; end
push!(bands,node); t0=time()
for b in bands; check!(S,b); S.nodes[b].kind==:open && auto_expand!(S,b;max_nodes=200_000,max_seconds=7200.0); end
println("cuts $(cuts): verdict ", session_verdict(S), " propagations ", S.n_checks, " nodes ", length(S.nodes), " wall ", round(time()-t0,digits=1), " s")
println(export_certificate(S, ARGS[2]))
