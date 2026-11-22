using VeryDiffAgentKernel, JSON; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","acas_prop16")); dead=JSON.parse(read("$d/dead.json",String))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>0.05))
r0=check!(S,1); println("root bound without lemmas: ", r0["distance_bound"])
node=1; total_lemma_nodes=0
for k in ["3","6","9"]
    nb=Int.(dead[k]["both_off"]); isempty(nb) && continue
    r=introduce_phase_lemma!(S,node,parse(Int,k),nb; net1="off", net2="off")
    p=prove_lemma!(S,r["lemma"]; max_nodes=3000, max_seconds=120.0)
    t=length(K.subtree(S,r["lemma_root"])); global total_lemma_nodes+=t
    c=check!(S,r["child"])
    println("layer $k: $(length(nb)) units off, lemma $(p["lemma_status"]) with $t proof nodes; bound now ", round(c["distance_bound"],digits=3))
    p["lemma_status"]=="proven" || break
    global node=r["child"]
end
a=auto_expand!(S,node; max_nodes=20000, max_seconds=300.0)
st=session_status(S)
println("verdict ", st["verdict"], "; main tree nodes ", st["main"]["nodes"], "; lemma proof nodes ", total_lemma_nodes, "; total ", st["total_nodes"])
println(export_certificate(S,joinpath(d,"chain9_cert.json")))
