# Bound rows: linear constraints on the shared noise symbols, collected during one propagation.
#
# All zonotopes of one propagation are affine functions of the same noise symbols ε ∈ [-1, 1]^m. Generator blocks
# have ids, and a block only gets new columns, so column c of block b is the same symbol at every layer. For each
# concrete input there is one ε that reproduces the values of all layers at the same time (the witness).
#
# A cached bound of a ReLU input is sound for every concrete input. Thus, if z_j = c + g·ε is the zonotope row of
# that ReLU input and u_j is its cached upper bound, then c + g·ε ≤ u_j holds at the witness of every concrete
# input. This row carries information when u_j < c + ‖g‖₁, i.e. when the cache is tighter than the zonotope
# (inherited caches, seeded bounds, earlier tightening). Lower bounds give rows in the same way.
#
# A row a·ε ≤ β tightens the bounds of a later unit u = c + g·ε by weak duality. For each λ ≥ 0:
#     u = c + (g − λa)·ε + λ(a·ε) ≤ c + ‖g − λa‖₁ + λβ.
# Any λ ≥ 0 gives a sound bound; the best λ is found exactly by a walk over sorted break points.
#
# Use: `BOUND_ROWS[] = K` (K > 0) makes `propagate!` keep the (at most) K rows with the largest relative gap per
# ReLU layer and use them to tighten the cached bounds of all later ReLU layers before their transformer runs.
# Off by default.

"""
    BoundRow(Z, terms, rhs)

The linear constraint Σ coefficient · variable ≤ rhs on the input `Z` of a ReLU layer. `terms` holds
(variable, neuron, coefficient) with variable 1 = Z₁, 2 = Z₂, 3 = ∂Z. `Z` is a reference to the live zonotope
(not a copy): if a caller rescales the noise symbols of the live zonotopes, the row stays correct.
"""
struct BoundRow
    Z::DiffZonotope
    terms::Vector{Tuple{Int,Int,Float64}}
    rhs::Float64
end

bound_row_variable(Z::DiffZonotope, v::Int) = v == 1 ? Z.Z₁ : v == 2 ? Z.Z₂ : Z.∂Z

"""Column offset of each generator block id over the zonotopes `Zs`, and the total number of noise symbols."""
function noise_layout(Zs)
    width = Dict{Int,Int}()
    for Z in Zs, (b, id) in enumerate(Z.generator_ids)
        width[id] = max(get(width, id, 0), size(Z.Gs[b], 2))
    end
    offset = Dict{Int,Int}()
    m = 0
    for id in sort(collect(keys(width)))
        offset[id] = m
        m += width[id]
    end
    return offset, m
end

"""Row j of zonotope Z as (c, g) over the layout `offset`, `m`."""
function noise_row(Z::Zonotope, j::Int, offset, m::Int)
    g = zeros(m)
    for (b, id) in enumerate(Z.generator_ids)
        G = Z.Gs[b]
        o = offset[id]
        @inbounds for col in 1:size(G, 2)
            g[o + col] += G[j, col]
        end
    end
    return Z.c[j], g
end

"""The row as a·ε ≤ β over the layout `offset`, `m`."""
function row_vector(r::BoundRow, offset, m::Int)
    a = zeros(m)
    c = 0.0
    for (v, j, coef) in r.terms
        cj, gj = noise_row(bound_row_variable(r.Z, v), j, offset, m)
        c += coef * cj
        a .+= coef .* gj
    end
    return a, r.rhs - c
end

"""
    dual_one_row(g, a, β)

min over λ ≥ 0 of ‖g − λa‖₁ + λβ. The function is convex and piecewise linear in λ, with break points g_i / a_i.
The slope at λ = 0 is β − Σ sign(g_i) a_i (for a_i ≠ 0, with sign(0) = −sign(a_i)). Each break point with
g_i / a_i > 0 adds 2|a_i| to the slope. Walk the sorted break points until the slope is not negative.
"""
function dual_one_row(g::Vector{Float64}, a::Vector{Float64}, β::Float64)
    base = sum(abs, g)
    isfinite(β) || return base
    slope = β
    breaks = Tuple{Float64,Float64}[]
    for i in eachindex(a)
        ai = a[i]
        ai == 0.0 && continue
        l = g[i] / ai
        if l > 0
            slope -= abs(ai)
            push!(breaks, (l, abs(ai)))
        else
            slope += abs(ai)
        end
    end
    slope >= 0 && return base            # λ = 0 is optimal: the row does not help
    sort!(breaks; by=first)
    λ = 0.0
    for (l, w) in breaks
        λ = l
        slope += 2w
        slope >= 0 && break
    end
    return min(base, sum(abs, g .- λ .* a) + λ * β)
end

"""Set an uninitialised cache to the fresh bounds of `Zin`; intersect an initialised one with them."""
function intersect_with_fresh!(bc::BoundsCache, Zin::DiffZonotope)
    b₁ = zono_bounds(Zin.Z₁); b₂ = zono_bounds(Zin.Z₂); ∂b = zono_bounds(Zin.∂Z)
    if !bc.initialized
        bc.lower₁ = b₁[:, 1]; bc.upper₁ = b₁[:, 2]
        bc.lower₂ = b₂[:, 1]; bc.upper₂ = b₂[:, 2]
        bc.∂lower = ∂b[:, 1]; bc.∂upper = ∂b[:, 2]
        bc.initialized = true
    else
        bc.lower₁ .= max.(bc.lower₁, b₁[:, 1]); bc.upper₁ .= min.(bc.upper₁, b₁[:, 2])
        bc.lower₂ .= max.(bc.lower₂, b₂[:, 1]); bc.upper₂ .= min.(bc.upper₂, b₂[:, 2])
        bc.∂lower .= max.(bc.∂lower, ∂b[:, 1]); bc.∂upper .= min.(bc.∂upper, ∂b[:, 2])
    end
end

"""
    dual_tighten!(bc, Zin, rows)

Tighten the cached Z₁, Z₂ and ∂Z bounds of the ReLU layer with input `Zin` with each row of `rows` (one row at a
time, the best row per bound). The cache is first intersected with the fresh bounds of `Zin`. The results are
rounded outward.
"""
function dual_tighten!(bc::BoundsCache, Zin::DiffZonotope, rows::Vector{BoundRow})
    intersect_with_fresh!(bc, Zin)
    isempty(rows) && return
    Zs = Zonotope[Zin.Z₁, Zin.Z₂, Zin.∂Z]
    for r in rows
        append!(Zs, (r.Z.Z₁, r.Z.Z₂, r.Z.∂Z))
    end
    offset, m = noise_layout(Zs)
    vecs = [row_vector(r, offset, m) for r in rows]
    for (Z, lo, hi) in ((Zin.Z₁, bc.lower₁, bc.upper₁), (Zin.Z₂, bc.lower₂, bc.upper₂), (Zin.∂Z, bc.∂lower, bc.∂upper))
        for j in eachindex(lo)
            c, g = noise_row(Z, j, offset, m)
            ng = .-g
            ub = Inf; lb = -Inf
            for (a, β) in vecs
                ub = min(ub, c + dual_one_row(g, a, β))
                lb = max(lb, c - dual_one_row(ng, a, β))
            end
            isfinite(ub) && (hi[j] = min(hi[j], nextfloat(ub + 1e-12 * abs(ub))))
            isfinite(lb) && (lo[j] = max(lo[j], prevfloat(lb - 1e-12 * abs(lb))))
        end
    end
end

"""
    infeasible_rows(rows; rtol=1e-9)

Test whether the rows have no common solution in the noise box, one row and pairs of rows at a time.
One row a·ε ≤ β: infeasible if min over the box of a·ε = −‖a‖₁ is > β.
Two rows: infeasible if min a₁·ε over {box, a₂·ε ≤ β₂} is > β₁. By weak duality, for each λ ≥ 0
    min a₁·ε ≥ −(‖−a₁ − λa₂‖₁ + λβ₂),
so the test is −dual_one_row(−a₁, a₂, β₂) > β₁ (a Farkas certificate with two multipliers 1 and λ). Both tests
need a margin of rtol·(1 + |β₁| + ‖a₁‖₁), so that rounding cannot make a feasible set look empty.
Return nothing, or (i, j) with the indices of the rows (j = i for a single row).
"""
function infeasible_rows(rows::Vector{BoundRow}; rtol::Float64=1e-9)
    isempty(rows) && return nothing
    Zs = Zonotope[]
    for r in rows
        append!(Zs, (r.Z.Z₁, r.Z.Z₂, r.Z.∂Z))
    end
    offset, m = noise_layout(Zs)
    vecs = [row_vector(r, offset, m) for r in rows]
    margin(a, β) = rtol * (1 + abs(β) + sum(abs, a))
    for (i, (a, β)) in enumerate(vecs)
        isfinite(β) || continue
        -sum(abs, a) > β + margin(a, β) && return (i, i)
    end
    for (i, (a₁, β₁)) in enumerate(vecs), (j, (a₂, β₂)) in enumerate(vecs)
        (i != j && isfinite(β₁) && isfinite(β₂)) || continue
        -dual_one_row(.-a₁, a₂, β₂) > β₁ + margin(a₁, β₁) && return (i, j)
    end
    return nothing
end

"""
    collect_bound_rows!(rows, bc, Zin, K, rtol)

Append to `rows` the cached bounds of the ReLU layer with input `Zin` that are tighter than the zonotope's own
bounds. The gap of a bound is its distance to the zonotope bound, relative to the width of the zonotope interval.
Only bounds with gap > `rtol` are candidates; the `K` largest gaps are kept (ties: variable, neuron, upper first).
"""
function collect_bound_rows!(rows::Vector{BoundRow}, bc::BoundsCache, Zin::DiffZonotope, K::Int, rtol::Float64)
    (K > 0 && bc.initialized) || return rows
    candidates = Tuple{Float64,Int,Int,Bool,Float64}[]          # (gap, variable, neuron, upper, bound)
    for (v, lo, hi) in ((1, bc.lower₁, bc.upper₁), (2, bc.lower₂, bc.upper₂), (3, bc.∂lower, bc.∂upper))
        zb = zono_bounds(bound_row_variable(Zin, v))
        for j in eachindex(lo)
            w = zb[j, 2] - zb[j, 1]
            w > 1e-12 || continue
            gap_up = (zb[j, 2] - hi[j]) / w
            isfinite(hi[j]) && gap_up > rtol && push!(candidates, (gap_up, v, j, true, hi[j]))
            gap_lo = (lo[j] - zb[j, 1]) / w
            isfinite(lo[j]) && gap_lo > rtol && push!(candidates, (gap_lo, v, j, false, lo[j]))
        end
    end
    sort!(candidates; by = x -> (-x[1], x[2], x[3], !x[4]))
    for (_, v, j, upper, b) in Iterators.take(candidates, K)
        push!(rows, upper ? BoundRow(Zin, [(v, j, 1.0)], b) : BoundRow(Zin, [(v, j, -1.0)], -b))
    end
    return rows
end
