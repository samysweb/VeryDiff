# Lookahead optimiser driving the kernel: at each open node, dry-run candidate expansions (VeryDiff's input
# split, other input dims, ReLU case splits on the top relu_scores units), each finished by VeryDiff, and keep
# the cheapest. Small subtrees are finished directly.
using VeryDiffAgentKernel, JSON, Printf; const K=VeryDiffAgentKernel
d=ARGS[1]; eps=parse(Float64,ARGS[2]); out=ARGS[3]
THR=parse(Int,get(ENV,"THR","12")); NIN=parse(Int,get(ENV,"NIN","3")); NRELU=parse(Int,get(ENV,"NRELU","4"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps))
function candidates(S,id)
    c=Any[]
    ss=split_scores(S,id;top_k=NIN)
    n=S.nodes[id]
    for e in ss["scores"]; push!(c,Dict("kind"=>"split","dim"=>e["dim"]))
        if get(ENV,"CUTS","0")=="1"
            lo,hi=n.region.lo[e["dim"]],n.region.hi[e["dim"]]
            for f in (0.25,0.75); push!(c,Dict("kind"=>"split","dim"=>e["dim"],"cut"=>lo+f*(hi-lo))); end
        end
    end
    rs=relu_scores(S,id;top_k=NRELU)
    for e in rs["top"]
        k,j=e["layer"],e["neuron"]
        x1=e["z1"][1]<0<e["z1"][2]; x2=e["z2"][1]<0<e["z2"][2]
        x1 && push!(c,Dict("kind"=>"split_relu","layer"=>k,"neuron"=>j,"var"=>"z1"))
        x2 && push!(c,Dict("kind"=>"split_relu","layer"=>k,"neuron"=>j,"var"=>"z2"))
        x1 && x2 && push!(c,Dict("kind"=>"phase4","layer"=>k,"neuron"=>j))
        e["d"][1]<0<e["d"][2] && push!(c,Dict("kind"=>"split_relu","layer"=>k,"neuron"=>j,"var"=>"d"))
    end
    return c
end
stats=Dict("relu"=>0,"input"=>0,"nondefault_input"=>0); t0=time()
queue=[1]
while !isempty(queue)
    id=popfirst!(queue); n=S.nodes[id]
    n.kind==:open || continue
    isnothing(n.last_check) && check!(S,id)
    S.nodes[id].kind==:open || continue
    cs=candidates(S,id)
    best=nothing; bestc=typemax(Int)
    for (i,a) in enumerate(cs)
        r=trial_actions!(S,id,[a]; baseline=false, max_nodes=bestc==typemax(Int) ? 10^6 : bestc)["results"][1]
        if get(r,"closed",false) && r["checks"]<bestc; best=a; bestc=r["checks"]; end
    end
    isnothing(best) && error("no candidate closed node $id")
    a=best; isrelu=a["kind"]!="split"
    isrelu ? (stats["relu"]+=1) : (stats["input"]+=1; a!=cs[1] && (stats["nondefault_input"]+=1))
    @printf("node %d depth %d: default %s best %s cost %d  [%.0fs, checks %d, trial %d]\n", id, n.depth, "-", JSON.json(a), bestc, time()-t0, S.n_checks, S.n_trial_propagations)
    kids=apply_action!(S,id,a)
    if bestc<=THR
        auto_expand!(S,id;max_nodes=10^6)
    else
        append!(queue,kids)
    end
end
st=session_status(S)
println("verdict ", session_verdict(S), " checks ", st["checks"], " nodes ", length(S.nodes), " inspection ", S.n_propagations-S.n_checks, " trial ", S.n_trial_propagations, " stats ", stats, " time ", round(time()-t0))
println(export_certificate(S,out))
