# Properties as adapters. Each obligation carries one adapter.
#
#   adapter_check(a, N1, N2, Zin, Zout, status) -> (sat, cex, heur_info, status′, distance_bound)
#   adapter_split_heuristic(a, Zin, Zout, heur_info, task) -> flat split index
#   needs_lp(a)::Bool
#   split_blocks(a)::Vector{Symbol}
#   stop_layer(a)  -> nothing (full propagation) or the ReLU layer index whose *input* is checked
#   check_tag(a)::String  (evidence tag)

abstract type PropertyAdapter end

# ---------------------------------------------------------------- ε-equivalence
struct EpsAdapter <: PropertyAdapter
    epsilon::Float64
    fn::Function
end
function EpsAdapter(epsilon::Float64)
    (isfinite(epsilon) && epsilon >= 0) || kerr("property", "epsilon must be finite and >= 0")
    return EpsAdapter(epsilon, VeryDiff.get_epsilon_property(epsilon))
end
adapter_check(a::EpsAdapter, N1, N2, Zin, Zout, status) = a.fn(N1, N2, Zin, Zout, status)
adapter_split_heuristic(a::EpsAdapter, Zin, Zout, heur, task) = VeryDiff.epsilon_split_heuristic(Zin, Zout, heur, task)
needs_lp(::EpsAdapter) = false
split_blocks(::EpsAdapter) = [:primary]
stop_layer(::EpsAdapter) = nothing
check_tag(::EpsAdapter) = "verydiff_eps"
property_json(a::EpsAdapter) = Dict{String,Any}("kind" => "epsilon", "epsilon" => a.epsilon)
"""Concrete semantics: does output pair (y1, y2) violate the property?"""
violates(a::EpsAdapter, y1, y2) = maximum(abs.(y1 .- y2)) > a.epsilon

# ---------------------------------------------------------------- lemma bounds
"""Statement of a ∂-bounds lemma at the *input* of ReLU layer `relu_layer`
(pre-activations). ∂ = N1 − N2. Optional per-network bounds n1/n2."""
struct LemmaStatement
    relu_layer::Int
    neurons::Vector{Int}
    dlo::Vector{Float64}
    dhi::Vector{Float64}
    n1lo::Union{Nothing,Vector{Float64}}
    n1hi::Union{Nothing,Vector{Float64}}
    n2lo::Union{Nothing,Vector{Float64}}
    n2hi::Union{Nothing,Vector{Float64}}
    terms::Vector{Tuple{String,Int,Float64}}   # relational part: Σ coeff·v ≤ rhs, v ∈ {d, z1, z2}[neuron]
    rhs::Float64
end
LemmaStatement(k, neurons, dlo, dhi, n1lo, n1hi, n2lo, n2hi) =
    LemmaStatement(k, neurons, dlo, dhi, n1lo, n1hi, n2lo, n2hi, Tuple{String,Int,Float64}[], Inf)

struct LemmaBoundsAdapter <: PropertyAdapter
    stmt::LemmaStatement
end

"""Maximum amount by which interval rows [l,u] exceed [lo,hi] (≤ 0 means contained)."""
function bound_excess(b::AbstractMatrix, rows, lo, hi)
    isnothing(lo) && return -Inf
    ex = -Inf
    for (k, r) in enumerate(rows)
        ex = max(ex, lo[k] - b[r, 1], b[r, 2] - hi[k])
    end
    return ex
end

function adapter_check(a::LemmaBoundsAdapter, N1, N2, Zin, Zk, status)
    s = a.stmt
    off, m = var_layout([Zk.∂Z, Zk.Z₁, Zk.Z₂])
    ex = -Inf; worst_row = nothing
    for (terms, rhs) in lemma_specs(s)
        row = combo_row(Zk, terms, off, m)
        e = row.c + sum(abs, row.g) - rhs
        if e > ex
            ex = e; worst_row = row
        end
    end
    if ex <= 0.0
        return true, nothing, nothing, nothing, ex
    end
    # Candidate (unverified) refutation points: the centre and the two input corners that push the worst claim.
    cands = Vector{Vector{Float64}}([copy(Zin.Z₁.c)])
    if !isnothing(worst_row) && !isempty(Zin.Z₁.Gs) && haskey(off, 1)
        G = Zin.Z₁.Gs[1]; o = off[1]
        w = sign.(worst_row.g[o+1:o+size(G, 2)])
        for c in (-1.0, 1.0)
            push!(cands, G * (c .* w) .+ Zin.Z₁.c)
        end
    end
    return false, nothing, (cands, nothing), nothing, ex
end

"""Row-restricted copy of `epsilon_split_heuristic`: only the lemma rows are weighted."""
function lemma_split_weights(Zk, rows)
    w = zeros(size(Zk.Z₁.influence[1], 1))
    for (g, inf) in zip(Zk.Z₁.Gs, Zk.Z₁.influence)
        w .+= sum(abs, abs.(@view g[rows, :]) * abs.(inf'), dims=1)[1, :]
    end
    for (g, inf) in zip(Zk.Z₂.Gs, Zk.Z₂.influence)
        w .+= sum(abs, abs.(@view g[rows, :]) * abs.(inf'), dims=1)[1, :]
    end
    return w
end
adapter_split_heuristic(a::LemmaBoundsAdapter, Zin, Zk, heur, task) = argmax(lemma_split_weights(Zk, a.stmt.neurons))
needs_lp(::LemmaBoundsAdapter) = false
split_blocks(::LemmaBoundsAdapter) = [:primary]
stop_layer(a::LemmaBoundsAdapter) = a.stmt.relu_layer
check_tag(::LemmaBoundsAdapter) = "lemma_bounds"
property_json(a::LemmaBoundsAdapter) = Dict{String,Any}("kind" => "lemma_bounds")

"""Concrete semantics for lemma refutation on hidden pre-activations h1, h2."""
function violates(a::LemmaBoundsAdapter, h1::AbstractVector, h2::AbstractVector)
    s = a.stmt
    d = h1[s.neurons] .- h2[s.neurons]
    (any(d .< s.dlo) || any(d .> s.dhi)) && return true
    if !isnothing(s.n1lo)
        (any(h1[s.neurons] .< s.n1lo) || any(h1[s.neurons] .> s.n1hi)) && return true
    end
    if !isnothing(s.n2lo)
        (any(h2[s.neurons] .< s.n2lo) || any(h2[s.neurons] .> s.n2hi)) && return true
    end
    if !isempty(s.terms)
        v = sum(a * (w == "d" ? h1[j] - h2[j] : w == "z1" ? h1[j] : h2[j]) for (w, j, a) in s.terms)
        v > s.rhs && return true
    end
    return false
end

"""Generic heuristic weight vector (used by the split_scores instrument)."""
function split_weights(a::EpsAdapter, Zin, Zout)
    w = zeros(size(Zout.Z₁.influence[1], 1))
    for (g, inf) in zip(Zout.Z₁.Gs, Zout.Z₁.influence)
        w .+= sum(abs, abs.(g) * abs.(inf'), dims=1)[1, :]
    end
    for (g, inf) in zip(Zout.Z₂.Gs, Zout.Z₂.influence)
        w .+= sum(abs, abs.(g) * abs.(inf'), dims=1)[1, :]
    end
    return w
end
split_weights(a::LemmaBoundsAdapter, Zin, Zk) = lemma_split_weights(Zk, a.stmt.neurons)

# ---------------------------------------------------------------- extension stubs
"""Top-1 / δ-Top-1 equivalence. Needs LP (Gurobi/GLPK) inside the kernel and
`verification_status` threading; not part of the MVP."""
struct Top1Adapter <: PropertyAdapter
    delta::Float64
end
"""Confidence robustness (TwoSafe): secondary split blocks :sec1/:sec2 via --input-epsilon."""
struct RobustnessAdapter <: PropertyAdapter
    delta::Float64
    input_epsilon::Float64
end
for T in (:Top1Adapter, :RobustnessAdapter)
    @eval begin
        adapter_check(a::$T, args...) = throw(NotImplementedError(string($T) * ".adapter_check"))
        adapter_split_heuristic(a::$T, args...) = throw(NotImplementedError(string($T) * ".adapter_split_heuristic"))
        needs_lp(::$T) = true
        stop_layer(::$T) = nothing
        check_tag(::$T) = throw(NotImplementedError(string($T)))
        violates(a::$T, args...) = throw(NotImplementedError(string($T) * ".violates"))
    end
end
property_json(a::Top1Adapter) = Dict{String,Any}("kind" => "top1", "delta" => a.delta)
property_json(a::RobustnessAdapter) = Dict{String,Any}("kind" => "robustness", "delta" => a.delta, "input_epsilon" => a.input_epsilon)
split_blocks(::Top1Adapter) = [:primary]
split_blocks(::RobustnessAdapter) = [:primary, :sec1, :sec2]

function make_property(p::AbstractDict)
    kind = string(get(p, "kind", ""))
    if kind == "epsilon"
        return EpsAdapter(Float64(p["epsilon"]))
    elseif kind == "top1"
        throw(NotImplementedError("Top-1 property (planned: Top1Adapter)"))
    elseif kind == "robustness"
        throw(NotImplementedError("confidence robustness (planned: RobustnessAdapter with :sec1/:sec2 split blocks)"))
    else
        kerr("property", "unknown property kind '$kind' (supported: epsilon)")
    end
end
