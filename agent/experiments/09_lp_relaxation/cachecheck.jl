using VeryDiffAgentKernel, Random; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=split_relu!(S,1,3,45;var="z1",t=0.975)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=1.0)["children"][1]
check!(S,n); bc=S.nodes[n].cache_out.bounds_cache
P=readlines(get(ENV,"PTS",joinpath(@__DIR__,"band_pts.txt"))); found=0; viol=Dict{String,Float64}()
for line in P
  x=parse.(Float64,split(line))
  e=eval_point(S,x)["hidden_preact"]; r=e["3"]["n1"][45]
  (0.975-1e-9<=r<=1.0+1e-9) || (println("point outside band ",r); continue)
  global found+=1
  for k in (3,6,9)
    h1=e[string(k)]["n1"]; h2=e[string(k)]["n2"]; c=bc[k]
    for (nm,v,lo,hi) in (("z1",h1,c.lower₁,c.upper₁),("z2",h2,c.lower₂,c.upper₂),("d",h1.-h2,c.∂lower,c.∂upper))
      w=maximum(max.(lo.-v,v.-hi)); w>1e-9 && (viol["L$k:$nm"]=max(get(viol,"L$k:$nm",0.0),w))
    end
  end
end
println("checked $found band points; cache violations (max amount): ", viol)
