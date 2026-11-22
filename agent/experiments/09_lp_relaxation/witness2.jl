using VeryDiffAgentKernel, JuMP, GLPK; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=parse(Bool,get(ENV,"FB","true")))
mode=get(ENV,"MODE","band")
n=1
if mode=="band"
  n=split_relu!(S,1,3,45;var="z1",t=0.975)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=1.0)["children"][1]
end
check!(S,n); nd=S.nodes[n]; res=K.inspect_propagation(S,nd)
x=parse.(Float64,split(readlines("$(ENV["PTS"])")[parse(Int,get(ENV,"PT","1"))]))
o1,o2=K.eval_raw(S,x)
Zs=Any[]; vals=Vector{Float64}[]; names=String[]
for L in K.get_layers(S.N), pos in (K.get_inputs(L)[1], K.get_outputs(L)[1])
  cz=res.ps.zono_storage.zonotopes[pos]; (isnothing(cz)||isnothing(cz.zonotope)) && continue
  nm=get(S.pos_names,pos,""); (haskey(o1,nm)) || continue
  v1=Float64.(vec(o1[nm])); v2=Float64.(vec(o2[nm]))
  for (tag,Z,v) in (("Z1",cz.zonotope.Z₁,v1),("Z2",cz.zonotope.Z₂,v2),("dZ",cz.zonotope.∂Z,v1.-v2))
    push!(Zs,Z); push!(vals,v); push!(names,"L$(L.layer_idx)$(pos==K.get_inputs(L)[1] ? "in" : "out"):$tag")
  end
end
off,m=K.var_layout(Zs)
md=Model(GLPK.Optimizer); set_silent(md); @variable(md,-1-1e-9<=e[1:m]<=1+1e-9); @variable(md, s>=0)
for (Z,v) in zip(Zs,vals), j in eachindex(v)
  r=K.affrow(Z,j,off,m); @constraint(md, r.c + r.g'*e - v[j] <= s); @constraint(md, -(r.c + r.g'*e - v[j]) <= s)
end
@objective(md,Min,s); optimize!(md)
println("mode $mode point $(get(ENV,"PT","1")): min max-residual of a single witness = ", objective_value(md), "  (", termination_status(md), ", m=$m)")
ew=value.(e)
# rows in the witness layout: rebuild relax rows with this (off, m)
rr,rv,spec=K.relax_rows(S,res,off,m)
viol=[(spec[i], rr[i]'*ew - rv[i]) for i in eachindex(rv) if rr[i]'*ew - rv[i] > 1e-7]
println(length(viol)," of ",length(rv)," relaxation rows violated at the witness")
for (sp,v) in sort(viol;by=x->-x[2])[1:min(end,15)]
  k,j,typ=sp; bc=res.cache.bounds_cache[k]
  println("  ",sp," by ",round(v,digits=6),"  z1∈[",round(bc.lower₁[j],digits=4),",",round(bc.upper₁[j],digits=4),"] z2∈[",round(bc.lower₂[j],digits=4),",",round(bc.upper₂[j],digits=4),"] d∈[",round(bc.∂lower[j],digits=4),",",round(bc.∂upper[j],digits=4),"]")
end
exit()
