using VeryDiffAgentKernel, JSON, Statistics; const K=VeryDiffAgentKernel
d=ARGS[1]; eps=parse(Float64,ARGS[2])
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps); fallback=true)
auto_expand!(S,1;max_nodes=10^6,max_seconds=7200.0)
sz=Dict{Int,Int}()
function size_of(id); n=S.nodes[id]; s=1+sum((size_of(c) for c in n.children); init=0); sz[id]=s; s; end
size_of(1)
rows=[(n.distance_bound-eps, sz[id], n.depth) for (id,n) in S.nodes if n.kind==:split]
open(ARGS[3],"w") do io; for r in rows; println(io, join(r,"\t")); end; end
x=[log(r[1]) for r in rows]; y=[log(r[2]) for r in rows]
b=cov(x,y)/var(x); a=mean(y)-b*mean(x); r=cor(x,y)
println("fit log(size) = $a + $b log(excess), r=$r, n=$(length(rows))")
for q in (0.01,0.03,0.1,0.2)
  sel=[r[2] for r in rows if q/1.5<r[1]<q*1.5]; isempty(sel) || println("excess≈$q: median size ", median(sel), " n=",length(sel))
end
