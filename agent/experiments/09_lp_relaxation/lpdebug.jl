using VeryDiffAgentKernel, JuMP, GLPK; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true, lp_relax=true)
n=split_relu!(S,1,3,45;var="z1",t=0.975)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=1.0)["children"][1]
check!(S,n); nd=S.nodes[n]; res=K.inspect_propagation(S,nd)
Zo=res.Zout
off,m=K.var_layout(vcat(K.lp_zonos(S,res,Int[],[Zo.∂Z];constraints=nd.constraints),K.relax_zonos(S,res)))
used=Int[]; spec=Any[]
A,r,_=K.lemma_constraints(S,res,Int[],off,m;constraints=nd.constraints,used_cons=used,relax=true,relax_spec=spec)
o=K.affrow(Zo.∂Z,4,off,m)
println("m=$m rows=$(size(A,1)) zono bound ", o.c+sum(abs,o.g))
model=Model(GLPK.Optimizer); set_silent(model)
@variable(model,-1<=e[1:m]<=1); cons=@constraint(model,A*e .<= r); @objective(model,Max,o.g'*e)
t=time(); optimize!(model); println("status ", termination_status(model), " primal ", objective_value(model)+o.c, " has_duals ", has_duals(model), " t=", time()-t)
if has_duals(model)
  dd=dual.(cons); for cand in (max.(0,-dd),max.(0,dd)); println(" dual bound ", K.dual_bound(o,A,r,cand)); end
end
feas(rows)=(md=Model(GLPK.Optimizer); set_silent(md); @variable(md,-1<=x[1:m]<=1); isempty(rows) || @constraint(md,A[rows,:]*x .<= r[rows]); optimize!(md); termination_status(md))
nc=size(A,1)-length(spec)
println("constraint rows only: ", feas(collect(1:nc)))
for typ in K.RELAX_TYPES
  idx=[nc+i for (i,s) in enumerate(spec) if s[3]==typ]
  println(typ, " (", length(idx), " rows) alone: ", feas(idx), "   with constraints: ", feas(vcat(1:nc, idx)))
end
for k in S.relu_layers
  idx=[nc+i for (i,s) in enumerate(spec) if s[1]==k]; println("layer $k rows: ", feas(idx))
end
