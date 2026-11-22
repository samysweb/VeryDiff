using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
check!(S,1)
for rep in 1:3
  t=time(); check!(S,1); println("check ", time()-t)
  t=time(); split_scores(S,1;top_k=2); println("split_scores ", time()-t)
  t=time(); relu_scores(S,1;top_k=4); println("relu_scores ", time()-t)
  t=time(); s=K.local_snapshot(S,1); println("snapshot ", time()-t)
  t=time(); kids=apply_action!(S,1,Dict("kind"=>"split_relu","layer"=>6,"neuron"=>69,"var"=>"z2")); for c in kids; check!(S,c); end; println("relu split + 2 checks ", time()-t)
  t=time(); K.local_restore!(S,1,s); println("restore ", time()-t)
  s=K.local_snapshot(S,1)
  t=time(); kids=apply_action!(S,1,Dict("kind"=>"split","dim"=>442)); for c in kids; check!(S,c); end; println("input split + 2 checks ", time()-t)
  K.local_restore!(S,1,s)
end
