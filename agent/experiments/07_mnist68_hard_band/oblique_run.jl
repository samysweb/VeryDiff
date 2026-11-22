using VeryDiffAgentKernel, JSON; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68")); A=parse(Float64,ARGS[1]); B=parse(Float64,ARGS[2]); qs=parse.(Float64,split(ARGS[3],","))
O=JSON.parse(read(ARGS[4],String)); α=Float64.(O["alpha"]); s0=O["s0"]
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=split_relu!(S,1,3,45;var="z1",t=A)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=B)["children"][1]
terms=[("z1",j,α[j]) for j in eachindex(α) if abs(α[j])>1e-12]
leaves=[n]
for q in qs   # chain of oblique cuts: Σ α z1 = c·x + s0
  t=O["smin"]+q*(O["smax"]-O["smin"])+s0
  r=split_lin!(S,leaves[end],3,terms,t); leaves[end]=r["children"][1]; push!(leaves,r["children"][2])
end
c0=S.n_checks
for l in leaves
  c=S.n_checks; auto_expand!(S,l;max_nodes=100000,max_seconds=3600.0); println("  piece: ", S.n_checks-c, " checks, closed=", is_closed(S,l)); flush(stdout)
end
println("band [$A,$B] with oblique cuts at $qs: total ", S.n_checks-c0, " checks, closed ", is_closed(S,n))
