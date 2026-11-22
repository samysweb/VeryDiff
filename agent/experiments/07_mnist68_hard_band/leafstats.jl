using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68")); A=parse(Float64,ARGS[1]); B=parse(Float64,ARGS[2])
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=split_relu!(S,1,3,45;var="z1",t=A)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=B)["children"][1]
auto_expand!(S,n;max_nodes=100000)
leaves=[x for x in values(S.nodes) if x.kind==:closed]
cnt=Dict{String,Int}(); for x in leaves; k=x.evidence["check"]; cnt[k]=get(cnt,k,0)+1; end
println("band [$A,$B]: checks ", S.n_checks, " leaves ", length(leaves), " by evidence ", cnt)
dims=Dict{Int,Int}(); for x in values(S.nodes); x.kind==:split && x.split_block==:primary && (dims[x.split_dim]=get(dims,x.split_dim,0)+1); end
println("input split dims (count): ", sort(collect(dims);by=x->-x[2]))
vols=[x.work_share for x in leaves if x.evidence["check"]!="infeasible"]
println("depth of feasible leaves: median ", sort([x.depth for x in leaves if x.evidence["check"]!="infeasible"])[end÷2])
