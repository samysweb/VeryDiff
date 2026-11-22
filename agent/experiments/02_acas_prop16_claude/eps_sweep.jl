using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","acas_prop16"))
for eps in (0.05, 0.02, 0.01, 0.005)
  S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps))
  check!(S,1)
  t=trial_lemma!(S,1,15,[49]; net1="off", lemma_max_nodes=100000, main_max_nodes=200000, max_seconds=900.0)
  println("eps=$eps: lemma ", t["lemma"]["proof_nodes"], " (", t["lemma"]["status"], "); main without ",
          t["without_lemma"]["subtree_nodes"], " (closed=", t["without_lemma"]["closed"], "), with ",
          t["with_lemma"]["subtree_nodes"], " (closed=", t["with_lemma"]["closed"], "); net ", t["net_saving_nodes"])
  flush(stdout)
end
