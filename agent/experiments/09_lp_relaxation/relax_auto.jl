using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68")); A=parse(Float64,ARGS[1]); B=parse(Float64,ARGS[2]); W=parse(Float64,get(ENV,"W","0"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true, lp_relax=get(ENV,"RELAX","1")=="1")
check!(S,1)
cuts= W>0 ? collect(A+W:W:B-W/2) : Float64[]
n=split_relu!(S,1,3,45;var="z1",t=A)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=B)["children"][1]
leaves=[n]; for t in cuts; r=split_relu!(S,leaves[end],3,45;var="z1",t=t); leaves[end]=r["children"][1]; push!(leaves,r["children"][2]); end
c0=S.n_checks; t0=time()
for l in leaves; auto_expand!(S,l;max_nodes=100000,max_seconds=7200.0); end
println("band [$A,$B] sub-band width $(W>0 ? W : B-A) relax=$(S.lp_relax): closed=$(is_closed(S,n)) propagations $(S.n_checks-c0), wall $(round(time()-t0,digits=1)) s")
if is_closed(S,n) && haskey(ENV,"OUT"); println(export_certificate(S,ENV["OUT"])); end
