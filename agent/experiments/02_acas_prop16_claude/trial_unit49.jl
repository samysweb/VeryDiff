using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","acas_prop16"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>0.05))
check!(S,1)
for (lbl, node) in [("whole box", 1)]
  t=trial_lemma!(S,node,15,[49]; net1="off", lemma_max_nodes=20000, main_max_nodes=20000, max_seconds=600.0)
  println(lbl, ": lemma ", t["lemma"], " with ", t["with_lemma"], " without ", t["without_lemma"], " net ", t["net_saving_nodes"])
end
c=split!(S,1,5; cut=0.0)["children"]; check!(S,c[1])
t=trial_lemma!(S,c[1],15,[49]; net1="off", lemma_max_nodes=20000, main_max_nodes=20000, max_seconds=600.0)
println("x5<0 half: lemma ", t["lemma"], " with ", t["with_lemma"], " without ", t["without_lemma"], " net ", t["net_saving_nodes"])
q=split!(S,c[1],5; cuts=[-0.375,-0.25])["children"]
for n in q
  check!(S,n)
  t=trial_lemma!(S,n,15,[49]; net1="off", lemma_max_nodes=20000, main_max_nodes=20000, max_seconds=600.0)
  println("cell x5 in [", S.nodes[n].region.lo[5], ",", S.nodes[n].region.hi[5], "]: lemma ", t["lemma"]["proof_nodes"], " (", t["lemma"]["status"], ") with ", t["with_lemma"]["subtree_nodes"], " without ", t["without_lemma"]["subtree_nodes"], " net ", t["net_saving_nodes"])
end
