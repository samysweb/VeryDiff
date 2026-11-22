using VeryDiffAgentKernel, JuMP, GLPK; const K=VeryDiffAgentKernel
d=get(ENV,"INST",joinpath(@__DIR__,"..","instances","mnist68"))
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true, lp_relax=true)
for (a,b) in ((0.975,1.0),(0.975,0.9875),(0.975,0.98125),(0.98,0.981))
  s0=K.local_snapshot(S,1)
  n=split_relu!(S,1,3,45;var="z1",t=a)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=b)["children"][1]
  check!(S,n); nd=S.nodes[n]; res=K.inspect_propagation(S,nd); Zo=res.Zout
  off,m=K.var_layout(vcat(K.lp_zonos(S,res,Int[],[Zo.∂Z];constraints=nd.constraints),K.relax_zonos(S,res)))
  used=Int[]; spec=Any[]; eqp=Tuple{Int,Int}[]
  A,r,_=K.lemma_constraints(S,res,Int[],off,m;constraints=nd.constraints,used_cons=used,relax=true,relax_spec=spec,eqpairs=eqp)
  o=K.affrow(Zo.∂Z,4,off,m)
  inp=falses(size(A,1)); for (i,k) in eqp; inp[i]=true; inp[k]=true; end; iq=findall(.!inp)
  md=Model(GLPK.Optimizer); set_silent(md); @variable(md,-1<=e[1:m]<=1)
  @constraint(md,A[iq,:]*e .<= r[iq]); for (i,k) in eqp; @constraint(md, A[i,:]'*e == (r[i]-r[k])/2); end
  @objective(md,Max,o.g'*e); optimize!(md)
  st=termination_status(md)
  println("band [$a,$b]: m=$m rows=$(size(A,1)) eq=$(length(eqp)) status=$st ", st==MOI.OPTIMAL ? "primal max $(objective_value(md)+o.c)" : "")
  K.local_restore!(S,1,s0)
end
