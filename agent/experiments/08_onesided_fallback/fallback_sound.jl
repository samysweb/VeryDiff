using VeryDiffAgentKernel, VeryDiff, JSON, Random; const K=VeryDiffAgentKernel
VeryDiff.ONE_SIDED_DIFF_FALLBACK[] = true
d="/home/user/VeryDiff/agent/tests/nets"
insts=[(joinpath(d,i["net1"]),joinpath(d,i["net2"]),joinpath(d,i["spec"]),i["epsilon"],i["name"]) for i in JSON.parse(read("$d/instances.json",String))]
push!(insts,(joinpath(@__DIR__,"..","instances","acas_prop16","net1.onnx"),joinpath(@__DIR__,"..","instances","acas_prop16","net2.onnx"),joinpath(@__DIR__,"..","instances","acas_prop16","spec.vnnlib"),0.05,"acas"))
push!(insts,(joinpath(@__DIR__,"..","instances","mnist68","net1.onnx"),joinpath(@__DIR__,"..","instances","mnist68","net2.onnx"),joinpath(@__DIR__,"..","instances","mnist68","spec.vnnlib"),1.0,"mnist"))
rng=MersenneTwister(1); bad=0; tot=0; moved=0
for (n1,n2,sp,eps,name) in insts
  S=open_session(n1,n2,sp,Dict("kind"=>"epsilon","epsilon"=>eps);fallback=true)
  auto_expand!(S,1;max_nodes=60)
  ids=[id for (id,n) in S.nodes if !isnothing(n.last_check)]
  for id in ids[1:min(25,end)]
    n=S.nodes[id]; res=K.inspect_propagation(S,n)
    bs=Dict(k=>(K.zono_bounds(z.zonotope.Z₁),K.zono_bounds(z.zonotope.Z₂),K.zono_bounds(z.zonotope.∂Z)) for (k,z) in enumerate(res.ps.zono_storage.zonotopes) if !isnothing(z) && !isnothing(z.zonotope))
    for _ in 1:40
      x=n.region.lo .+ rand(rng,length(n.region.lo)).*(n.region.hi .- n.region.lo)
      o1,o2=K.eval_raw(S,x)
      for (pos,(b1,b2,bd)) in bs
        haskey(S.pos_names,pos) || continue
        nm=S.pos_names[pos]; haskey(o1,nm) || continue
        y1=Float64.(vec(o1[nm])); y2=Float64.(vec(o2[nm])); global tot+=1
        ok=all(b1[:,1].-1e-7 .<= y1 .<= b1[:,2].+1e-7) && all(b2[:,1].-1e-7 .<= y2 .<= b2[:,2].+1e-7) && all(bd[:,1].-1e-7 .<= y1.-y2 .<= bd[:,2].+1e-7)
        ok || (global bad+=1; bad<5 && println("VIOLATION $name node $id pos $pos"))
      end
    end
  end
  println(name, ": verdict after 60 ", session_verdict(S))
end
println("checked tensor samples: $tot, violations: $bad")
