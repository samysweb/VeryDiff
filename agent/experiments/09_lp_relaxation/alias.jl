using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0))
res=K.run_propagation(S,S.root_box,K.TaskBounds(),Int[],nothing)
seen=Dict{Ptr{Float64},Vector{String}}()
for (pos,cz) in enumerate(res.ps.zono_storage.zonotopes)
  (isnothing(cz) || isnothing(cz.zonotope)) && continue
  for (nm,Z) in (("Z1",cz.zonotope.Z₁),("Z2",cz.zonotope.Z₂),("dZ",cz.zonotope.∂Z))
    push!(get!(seen,pointer(Z.c),String[]),"pos$pos.$nm.c")
    for (bi,G) in enumerate(Z.Gs); size(G,2)==0 && continue; push!(get!(seen,pointer(G),String[]),"pos$pos.$nm.G[$bi]"); end
  end
end
shared=[v for v in values(seen) if length(v)>1]
println(length(shared)," shared buffers"); for v in shared[1:min(end,15)]; println("  ",v); end
