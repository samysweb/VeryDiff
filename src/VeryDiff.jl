module VeryDiff

#using MaskedArrays
using LinearAlgebra
#using SparseArrays
using VNNLib
#using ThreadPinning

const NEW_HEURISTIC = Ref{Bool}(true)

const USE_DIFFZONO = Ref{Bool}(true)

"""If true, a ReLU unit whose fresh zonotope bounds give the any_any case but whose (tighter) cached bounds give a
one-sided case (any_neg, neg_any, any_pos, pos_any) keeps the d-based any_any relaxation, with the cached d bounds.
Tightening the cache (splits, lemmas, inherited bounds) then cannot make the differential relaxation looser than
with the fresh bounds. Sound for any phase combination, since min(0,d) <= ReLU(z1) - ReLU(z2) <= max(0,d).
Off by default (VeryDiff's behaviour is unchanged)."""
const ONE_SIDED_DIFF_FALLBACK = Ref{Bool}(false)

function __init__()
    BLAS.set_num_threads(1)
end

#pinthreads(:cores)

const FIRST_ROUND = Ref{Bool}(true)

include("Util/simd_bool.jl")
include("Debugger/Debugger.jl")
include("Definitions/Definitions.jl")
using .Definitions

include("Transformers/Transformers.jl")
using .Transformers

include("MultiThreadding.jl")

include("Properties/Properties.jl")
using .Properties

include("Verifier.jl")
include("Cli.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export parse_network
export Zonotope, DiffZonotope, PropState
export zono_optimize, zono_bounds
export verify_network
export get_epsilon_property, epsilon_split_heuristic, get_epsilon_property_naive
export get_top1_property, top1_configure_split_heuristic

end # module AlphaZono
