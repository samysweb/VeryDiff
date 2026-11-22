using VeryDiffAgentKernel, JSON; const K=VeryDiffAgentKernel
d="/home/user/VeryDiff/agent/tests/nets"; I=Dict(i["name"]=>i for i in JSON.parse(read("$d/instances.json",String)))
for name in ("safe_hard","safe_mid"), relax in (false,true)
  i=I[name]; S=open_session(joinpath(d,i["net1"]),joinpath(d,i["net2"]),joinpath(d,i["spec"]),Dict("kind"=>"epsilon","epsilon"=>i["epsilon"]); fallback=true, lp_relax=relax)
  t=time(); auto_expand!(S,1;max_nodes=parse(Int,get(ENV,"CAP","3000")),max_seconds=120.0)
  println("$name relax=$relax: ", S.n_checks, " checks in ", round(time()-t,digits=1), " s, verdict ", session_verdict(S)); flush(stdout)
end
