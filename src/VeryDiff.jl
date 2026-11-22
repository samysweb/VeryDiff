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

"""Rule for ONE_SIDED_DIFF_FALLBACK. false: fall back when the fresh zonotope bounds give any_any (the rule above).
true: fall back for each unit in a one-sided case (under the cached bounds) whose d-based relaxation gives a
narrower ∂ output row (smaller radius) than the one-sided relaxation. This rule depends only on the cached bounds
and the input zonotope, not on how the zonotope's fresh bounds compare with the cache. Off by default."""
const NARROWER_DIFF_FALLBACK = Ref{Bool}(false)

"""Number of bound rows that `propagate!` keeps per ReLU layer (see Transformers/BoundRows.jl): cached bounds
that are tighter than the zonotope's own bounds, used as linear constraints to tighten the cached bounds of later
ReLU layers by weak duality. 0 = off (default; VeryDiff's behaviour is unchanged)."""
const BOUND_ROWS = Ref{Int}(0)
"""Minimum gap (relative to the width of the zonotope interval) for a cached bound to become a bound row."""
const BOUND_ROWS_RTOL = Ref{Float64}(0.05)

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
