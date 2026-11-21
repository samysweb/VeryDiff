# Optional: build a sysimage to cut kernel start-up/JIT latency.
#   julia --project=agent/julia/VeryDiffAgentKernel agent/julia/VeryDiffAgentKernel/build_sysimage.jl [out.so]
# Then set VDA_SYSIMAGE=<out.so> for the MCP server.
# PackageCompiler is a dependency of VeryDiff's own project; stack that environment.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "..")))
using PackageCompiler
out = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "vdagent_sysimage.so")
script = tempname() * ".jl"
write(script, "using VeryDiffAgentKernel; VeryDiffAgentKernel.warmup()\n")
create_sysimage([:VeryDiffAgentKernel]; sysimage_path=out, precompile_execution_file=script)
println("sysimage: ", out)
