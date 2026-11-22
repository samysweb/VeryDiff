# Subgoal r ∈ [A,B]: enumerate layer-1 activation regions by phase splits on unstable layer-1 units (z1 at
# ReLU layer 3), then let VeryDiff discharge each region.
using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d="/tmp/claude-0/mnist"; A=parse(Float64,ARGS[1]); B=parse(Float64,ARGS[2]); DEPTH=parse(Int,get(ENV,"DEPTH","11"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=1
A>0.06 && (n=split_relu!(S,n,3,45;var="z1",t=A)["children"][2]); B<1.13 && (n=split_relu!(S,n,3,45;var="z1",t=B)["children"][1])
c0=S.n_checks; stats=Dict("regions"=>0,"infeasible"=>0,"closed_by_check"=>0,"auto"=>0)
function unstable1(id)
    bc=S.nodes[id].cache_out.bounds_cache[3]
    [j for j in eachindex(bc.lower₁) if j!=45 && bc.lower₁[j]<0<bc.upper₁[j]]
end
function solve(id, depth)
    x=check!(S,id)
    if S.nodes[id].kind==:closed
        S.nodes[id].evidence["check"]=="infeasible" ? (stats["infeasible"]+=1) : (stats["closed_by_check"]+=1); return
    end
    us=unstable1(id)
    if isempty(us) || depth==0
        stats["regions"]+=1; c=S.n_checks; auto_expand!(S,id;max_nodes=100000,max_seconds=3600.0); stats["auto"]+=S.n_checks-c; return
    end
    bc=S.nodes[id].cache_out.bounds_cache[3]
    j=us[argmax([min(-bc.lower₁[j],bc.upper₁[j]) for j in us])]       # most balanced unstable unit
    for c in split_relu!(S,id,3,j;var="z1")["children"]; solve(c, depth-1); end
end
t=time(); solve(n, DEPTH)
println("band [$A,$B]: closed=$(is_closed(S,n)) checks $(S.n_checks-c0) ", stats, " $(round(time()-t))s")
