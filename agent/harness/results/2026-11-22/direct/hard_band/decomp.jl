using VeryDiffAgentKernel; const K=VeryDiffAgentKernel
d="/tmp/claude-0/mnist"
S=open_session("$d/net1.onnx","$d/net2.onnx","$d/spec.vnnlib",Dict("kind"=>"epsilon","epsilon"=>1.0); fallback=true)
n=split_relu!(S,1,3,45;var="z1",t=0.975)["children"][2]; n=split_relu!(S,n,3,45;var="z1",t=1.0)["children"][1]
check!(S,n); println("band [0.975,1.0] bound ", S.nodes[n].distance_bound)
res=K.inspect_propagation(S,S.nodes[n]); DZ=res.Zout.∂Z; row=4
cols=K.relaxation_columns(S,res); owner=Dict((gid,c)=>(k,v,j) for (k,v,j,gid,c) in cols)
mass=Dict{String,Float64}()
for (bi,gid) in enumerate(collect(DZ.generator_ids)), c in 1:size(DZ.Gs[bi],2)
    g=abs(DZ.Gs[bi][row,c]); g==0 && continue
    key=haskey(owner,(gid,c)) ? (o=owner[(gid,c)]; "ReLU$(o[1]):$(o[2])") : "input/other(block $gid)"
    mass[key]=get(mass,key,0.0)+g
end
println("centre ", DZ.c[row], "  total generator mass ", sum(values(mass)))
for (k,v) in sort(collect(mass);by=x->-x[2]); println("  ",k," ",round(v,digits=4)); end
# LP bound for the row with band rows
ok,ev,w=K.lp_fallback(S,S.nodes[n],S.obligations[1],res); println("LP bound ", 1+w)
