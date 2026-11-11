using Test

using VNNLib
using VNNLib.OnnxParser: Node, ONNXLinear, ONNXRelu, OnnxNet, ONNXAddConst

using VeryDiff
using VeryDiff.Definitions: executable_network

VeryDiff.USE_DIFFZONO[] = true
VeryDiff.NEW_HEURISTIC[] = true

include("unit/util/simd_bool.jl")

@testset "VeryDiff Propagation" begin
    include("unit/propagation/main.jl")
end
@testset "VeryDiff Properties" begin
    include("unit/properties/main.jl")
end

include("fuzz.jl")
