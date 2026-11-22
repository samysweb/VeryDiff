# Best-first proof search with a one-step proxy: at each open node, try VeryDiff's input split (and the next
# NIN-1 dims) and single ReLU splits on the top NRELU relu_scores units (z1 and z2), check the children, and keep the
# action whose children have the smallest total excess over ε. Closed children count as excess 0.
using VeryDiffAgentKernel, JSON, Printf, DataStructures; const K=VeryDiffAgentKernel
d=ARGS[1]; eps=parse(Float64,ARGS[2]); out=ARGS[3]
NIN=parse(Int,get(ENV,"NIN","2")); NRELU=parse(Int,get(ENV,"NRELU","4")); MAXN=parse(Int,get(ENV,"MAXN","400000"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps); fallback=get(ENV,"FALLBACK","1")=="1")
# estimated subtree size from the bound excess (table calibrated on VeryDiff's own tree, log-log interpolation)
const TAB=[(parse(Float64,a),parse(Float64,b)) for (a,b) in split.(readlines(get(ENV,"COSTTAB",joinpath(d,"costtab.tsv"))),'\t')]
function f(e)
    e<=TAB[1][1] && return TAB[1][2]
    for i in 2:length(TAB)
        if e<=TAB[i][1]
            (e0,s0),(e1,s1)=TAB[i-1],TAB[i]
            return exp(log(s0)+(log(s1)-log(s0))*(log(e)-log(e0))/(log(e1)-log(e0)))
        end
    end
    (e0,s0),(e1,s1)=TAB[end-1],TAB[end]
    return exp(log(s1)+(log(s1)-log(s0))*(log(e)-log(e1))/(log(e1)-log(e0)))
end
excess(id)=(n=S.nodes[id]; n.kind==:closed ? 1.0 : f(max(n.distance_bound-eps,1e-9)))
function cands(id)
    c=Any[]
    for e in split_scores(S,id;top_k=NIN)["scores"]; push!(c,Dict("kind"=>"split","dim"=>e["dim"])); end
    if NRELU>0
        for e in relu_scores(S,id;top_k=NRELU)["top"]
            e["z1"][1]<0<e["z1"][2] && push!(c,Dict("kind"=>"split_relu","layer"=>e["layer"],"neuron"=>e["neuron"],"var"=>"z1"))
            e["z2"][1]<0<e["z2"][2] && push!(c,Dict("kind"=>"split_relu","layer"=>e["layer"],"neuron"=>e["neuron"],"var"=>"z2"))
        end
    end
    return c
end
check!(S,1)
heap=BinaryMaxHeap{Tuple{Float64,Int}}(); push!(heap,(S.nodes[1].distance_bound,1))
t0=time(); nrelu=0; ninput=0; nalt=0; explore=0; it=0
while !isempty(heap) && length(S.nodes)<MAXN
    global explore, ninput, nalt, nrelu, it
    _,id=pop!(heap); S.nodes[id].kind==:open || continue
    cs=cands(id); best=nothing; bestx=Inf
    for (i,a) in enumerate(cs)
        snap=K.local_snapshot(S,id)
        kids=apply_action!(S,id,a); for c in kids; check!(S,c); end
        x=sum(excess,kids); K.local_restore!(S,id,snap)
        (x<bestx-1e-12) && (best=a; bestx=x)
    end
    explore+=2*length(cs)
    a=best; a["kind"]=="split" ? (ninput+=1; a!=cs[1] && (nalt+=1)) : (nrelu+=1)
    for c in apply_action!(S,id,a); check!(S,c); S.nodes[c].kind==:open && push!(heap,(S.nodes[c].distance_bound,c)); end
    it+=1
    (it%200==0 || it<=5) && @printf("%d expansions, nodes %d, open %d, relu %d, input %d (alt %d), top bound %.4f, %.0fs\n", it, length(S.nodes), length(heap), nrelu, ninput, nalt, isempty(heap) ? 0 : first(heap)[1], time()-t0); flush(stdout)
end
cert_props=count(n->!isnothing(n.cache_out), values(S.nodes))
println("verdict ", session_verdict(S), " nodes ", length(S.nodes), " certificate propagations ", cert_props, " relu splits ", nrelu, " input ", ninput, " (alt dims ", nalt, ") exploration≈", explore, " trial ", S.n_trial_propagations, " time ", round(time()-t0))
session_verdict(S)=="SAFE" && println(export_certificate(S,out))
