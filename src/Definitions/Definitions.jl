module Definitions

using LinearAlgebra
using Statistics

using VNNLib
using VNNLib.OnnxParser: Node, ONNXLinear, ONNXRelu, ONNXAddConst

using VeryDiff

include("Network.jl")
include("SortedVector.jl")
include("AbstractDomains.jl")
include("DataStructures.jl")
include("PropState.jl")
include("Zonotope.jl")

export Network,GeminiNetwork,Layer,ZeroDense,DiffLayer, get_input_indices, get_zonos_at_pos, executable_network
export Zonotope,DiffZonotope,BoundsCache,CachedZonotope,ZonotopeStorage
export resize_zonotope_storage!
export VerificationTask, PropState, reset_ps!, first_pass, get_zonotope, get_layer, get_zonotope!, get_free_generator_id!
export SortedVector, union, intersect_indices, find_index_position, attempt_find_index_position
export parse_network, get_layers, get_inputs, get_outputs, get_layer1, get_diff_layer, get_layer2
export configure_first_usage!, prepare_prop_state!, zonos_initialized
export updateGenerators!, updateGeneratorsMul!, updateGeneratorsAdd!, updateGeneratorsAddMul!, updateGeneratorsSub!, updateGeneratorsSubMul!
export zono_optimize, zono_bounds, zono_get_max_vector

end