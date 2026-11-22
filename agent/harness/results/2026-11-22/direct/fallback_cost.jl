using VeryDiffAgentKernel, VeryDiff; const K=VeryDiffAgentKernel
for (d,eps) in (("/tmp/claude-0/acas",0.05),("/tmp/claude-0/mnist",1.0))
  for flag in (true,)
    VeryDiff.ONE_SIDED_DIFF_FALLBACK[] = flag
    S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>eps);fallback=flag)
    check!(S,1); rb=S.nodes[1].distance_bound
    mv=0
    t=time(); auto_expand!(S,1;max_nodes=10^6,max_seconds=7200.0)
    println(basename(d)," flag=$flag root bound $(round(rb,digits=4)) movable rows at root $mv  checks ",S.n_checks," ",session_verdict(S)," ",round(time()-t),"s")
  end
end
