# Proof tree data structures and *derived* closure.

mutable struct Node
    id::Int
    obligation::Int
    parent::Int                       # 0 for an obligation root
    side::Int                         # 0 root / assume-child, 1 lower child, 2 upper child
    region::Box
    kind::Symbol                      # :open | :split | :assume | :closed
    split_block::Symbol               # :primary | :sec1 | :sec2 (MVP: :primary only)
    split_dim::Int
    split_cut::Float64
    lemma::Int                        # for :assume
    evidence::Union{Nothing,Dict{String,Any}}
    children::Vector{Int}
    lemmas_in_scope::Vector{Int}
    last_check::Union{Nothing,Dict{String,Any}}
    rationale::String
    created_by::Symbol                # :agent | :auto
    cache_in::TaskBounds              # owned deepcopy, inherited from parent (frozen)
    cache_out::Union{Nothing,TaskBounds}  # owned deepcopy after this node's own propagation
    verification_status::Any
    work_share::Float64               # volume fraction within its obligation's root
    depth::Int
    distance_bound::Float64           # last known (Inf if never checked)
    cex::Bool
end

function Node(id, obligation, parent, side, region, lemmas, cache_in, work_share, depth, created_by;
              distance_bound=Inf, verification_status=nothing)
    return Node(id, obligation, parent, side, region, :open, :primary, 0, NaN, 0, nothing, Int[],
        copy(lemmas), nothing, "", created_by, cache_in, nothing, deepcopy(verification_status),
        work_share, depth, distance_bound, false)
end

is_leaf(n::Node) = n.kind == :open || n.kind == :closed
"""Cache a child inherits: the node's own post-propagation cache if it has one, else its inherited cache."""
inheritable_cache(n::Node) = isnothing(n.cache_out) ? n.cache_in : n.cache_out

mutable struct Obligation
    id::Int
    kind::Symbol                      # :main | :lemma
    root::Int
    lemma::Int                        # 0 for :main
    adapter::PropertyAdapter
end

mutable struct Lemma
    id::Int
    stmt::LemmaStatement
    scope_node::Int                   # the :assume node
    obligation::Int                   # its proof obligation
    refuted::Bool
    refutation::Union{Nothing,Dict{String,Any}}
    rationale::String
end
