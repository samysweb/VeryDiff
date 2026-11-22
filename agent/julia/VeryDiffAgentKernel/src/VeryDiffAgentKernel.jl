"""
VeryDiffAgentKernel — LCF-style trusted kernel that lets an untrusted agent drive
VeryDiff's proof search. All proof state lives here; every proof action is checked
here; closure/SAFE is derived from the tree, never asserted.
"""
module VeryDiffAgentKernel

using LinearAlgebra
using Statistics
using SHA
using JSON
using CodecZlib
using DataStructures

using VNNLib
using VNNLib.OnnxParser: ONNXRelu, ONNXSoftmax, compute_all_outputs

using VeryDiff
using VeryDiff.Definitions
using VeryDiff.Definitions: VerificationTask, PropState, BoundsCache, DiffLayer, TaskBounds,
    get_layers, get_inputs, get_outputs, get_zonos_at_pos, first_pass,
    configure_first_usage!, zonos_initialized, prepare_prop_state!,
    resize_zonotope_storage!, get_zonotope, zono_bounds, sort_network
using VeryDiff.Transformers: init_layer!, propagate_layer!, get_selectors

const KERNEL_VERSION = "0.1.0"
const CERT_VERSION = 1

struct KernelError <: Exception
    type::String
    msg::String
end
Base.showerror(io::IO, e::KernelError) = print(io, "KernelError(", e.type, "): ", e.msg)
kerr(type, msg) = throw(KernelError(type, msg))
require(cond, msg; type="precondition") = cond || kerr(type, msg)

struct NotImplementedError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NotImplementedError) = print(io, "NotImplemented: ", e.msg)

include("Regions.jl")
include("Props.jl")
include("ProofTree.jl")
include("Session.jl")
include("Kernel.jl")
include("Eval.jl")
include("Actions.jl")
include("AutoExpand.jl")
include("Lemmas.jl")
include("Inspect.jl")
include("Certificate.jl")
include("RPC.jl")

export Box, open_session, check!, split!, auto_expand!, introduce_lemma!, introduce_phase_lemma!, prove_lemma!,
    retract_lemma!, refute_lemma!, prune!, submit_counterexample!, export_certificate,
    session_status, list_layers, list_open_leaves, get_node, inspect_layer, output_bounds,
    split_scores, cache_bounds, eval_point, is_closed, session_verdict, serve

end
