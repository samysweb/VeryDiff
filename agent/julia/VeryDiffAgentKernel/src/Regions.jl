# Exact box regions and their (coverage-preserving) conversion to VeryDiff tasks.

struct Box
    lo::Vector{Float64}
    hi::Vector{Float64}
    function Box(lo::Vector{Float64}, hi::Vector{Float64})
        length(lo) == length(hi) || kerr("region", "lo/hi length mismatch")
        all(isfinite, lo) && all(isfinite, hi) || kerr("region", "non-finite box bounds")
        all(lo .<= hi) || kerr("region", "box with lo > hi")
        return new(copy(lo), copy(hi))
    end
end

Base.:(==)(a::Box, b::Box) = a.lo == b.lo && a.hi == b.hi
dim(b::Box) = length(b.lo)

"""Children of a split at `cut` along `d`: [lo, cut] and [cut, hi]. Both share the identical Float64 `cut`."""
function split_box(b::Box, d::Int, cut::Float64)
    (b.lo[d] < cut < b.hi[d]) || kerr("split", "cut $cut not strictly inside ($(b.lo[d]), $(b.hi[d])) on dim $d")
    hi1 = copy(b.hi); hi1[d] = cut
    lo2 = copy(b.lo); lo2[d] = cut
    return Box(copy(b.lo), hi1), Box(lo2, copy(b.hi))
end

"""Volume of `b` relative to `root`, over the dimensions that are non-degenerate in `root`."""
function volume_fraction(b::Box, root::Box)
    v = 1.0
    for i in eachindex(b.lo)
        w = root.hi[i] - root.lo[i]
        w > 0 && (v *= (b.hi[i] - b.lo[i]) / w)
    end
    return v
end

contains_point(b::Box, x::AbstractVector{<:Real}) = length(x) == dim(b) && all(b.lo .<= x .<= b.hi)

"""
Centre/radius representation of `b` that covers `b` exactly in real arithmetic.

Bounds are first widened outward by one ulp (`prevfloat`/`nextfloat`); the radius is
then rounded up (`nextfloat`) so that, in exact arithmetic, `[mid-dist, mid+dist] ⊇ [lo, hi]`.
Coverage is asserted with exact rational arithmetic. Dimensions in which the root box is
degenerate (lo == hi) are passed as fixed centre values (as VeryDiff itself does).
"""
function center_radius(b::Box, active_dims::Vector{Int})
    mid = copy(b.lo)
    dist = Float64[]
    for i in active_dims
        lo = prevfloat(b.lo[i]); hi = nextfloat(b.hi[i])
        m = lo / 2 + hi / 2
        d = max(nextfloat(m - lo), nextfloat(hi - m))
        mid[i] = m
        push!(dist, d)
        # exact coverage check (cheap: one rational comparison per dimension)
        R(x) = Rational{BigInt}(x)
        (R(m) - R(d) <= R(b.lo[i]) && R(m) + R(d) >= R(b.hi[i])) ||
            kerr("internal", "outward conversion failed to cover dim $i")
    end
    for i in setdiff(1:dim(b), active_dims)
        b.lo[i] == b.hi[i] || kerr("internal", "inactive dim $i is not degenerate")
    end
    return mid, dist
end

"""Build a VeryDiff `VerificationTask` for box `b` with the given bounds cache.
`work_share` is always 1.0 because the kernel uses a fresh PropState per propagation."""
function to_task(b::Box, active_dims::Vector{Int}, task_bounds::TaskBounds; verification_status=nothing)
    mid, dist = center_radius(b, active_dims)
    return VerificationTask(mid, dist, copy(active_dims),
        nothing, nothing, nothing, nothing,
        verification_status, Inf, 1.0, task_bounds)
end
