# Concrete evaluation (cex re-evaluation and lemma refutation).
#
# "Output" means the tensor VeryDiff verifies: the network output, or the input of the
# final Softmax if that was removed (as verify_network does). Hidden pre-activations are
# the inputs of each ReLU layer, available for both networks via compute_all_outputs.

function eval_raw(S::Session, x::Vector{Float64})
    length(x) == dim(S.root_box) || kerr("eval", "point has dimension $(length(x)), expected $(dim(S.root_box))")
    all(isfinite, x) || kerr("eval", "point has non-finite entries")
    o1 = compute_all_outputs(S.E1, x)
    o2 = compute_all_outputs(S.E2, x)
    return o1, o2
end

"""Returns Dict with y1, y2 (verified output), dist (max |y1-y2|) and per-ReLU-layer pre-activations."""
function eval_point(S::Session, x::Vector{Float64}; hidden::Bool=true)
    o1, o2 = eval_raw(S, x)
    y1 = Float64.(vec(o1[S.out_tensor])); y2 = Float64.(vec(o2[S.out_tensor]))
    r = Dict{String,Any}("y1" => y1, "y2" => y2, "dist" => maximum(abs.(y1 .- y2)),
        "in_spec_box" => contains_point(S.root_box, x))
    if hidden
        h = Dict{String,Any}()
        for k in S.relu_layers
            name = S.pos_names[get_inputs(layer_by_idx(S, k))[1]]
            h[string(k)] = Dict("n1" => Float64.(vec(o1[name])), "n2" => Float64.(vec(o2[name])))
        end
        r["hidden_preact"] = h
    end
    return r
end

"""Pre-activations of ReLU layer k for both networks."""
function hidden_at(S::Session, x::Vector{Float64}, k::Int)
    o1, o2 = eval_raw(S, x)
    name = S.pos_names[get_inputs(layer_by_idx(S, k))[1]]
    return Float64.(vec(o1[name])), Float64.(vec(o2[name]))
end

"""Is `x` a genuine counterexample to the main property? (in spec box + concrete violation)"""
function genuine_cex(S::Session, x::Vector{Float64})
    contains_point(S.root_box, x) || return false, nothing
    r = eval_point(S, x; hidden=false)
    return violates(S.property, r["y1"], r["y2"]), r
end

"""Does `x` (inside the lemma's scope region) concretely violate the lemma?"""
function refutes_lemma(S::Session, L::Lemma, x::Vector{Float64})
    region = S.nodes[L.scope_node].region
    contains_point(region, x) || return false
    h1, h2 = hidden_at(S, x, L.stmt.relu_layer)
    return violates(LemmaBoundsAdapter(L.stmt), h1, h2)
end

clamp_to(b::Box, x::Vector{Float64}) = clamp.(x, b.lo, b.hi)
