using VeryDiff
import VeryDiff: VerificationTask, PropState, prepare_prop_state!, propagate!, GeminiNetwork, zono_bounds
import VeryDiff.Transformers: dual_one_row, collect_bound_rows!, BoundRow
using VNNLib.OnnxParser: ONNXRelu
using Test
using Random

@testset "Bound rows" begin
    test_seed = "VERYDIFF_TEST_SEED" in keys(ENV) ? parse(Int, ENV["VERYDIFF_TEST_SEED"]) : rand(1:999999)
    Random.seed!(test_seed)
    @info "Bound rows test seed: $(test_seed)"

    @testset "dual_one_row is the minimum over λ ≥ 0" begin
        φ(g, a, β, λ) = sum(abs, g .- λ .* a) + λ * β
        for _ in 1:500
            m = rand(1:12)
            g = randn(m); a = randn(m)
            rand() < 0.3 && (a[rand(1:m)] = 0.0)
            β = randn() * 2
            v = dual_one_row(g, a, β)
            # the minimum of a convex piecewise linear function is at λ = 0 or at a break point g_i / a_i ≥ 0
            cands = [0.0; [g[i] / a[i] for i in 1:m if a[i] != 0.0 && g[i] / a[i] > 0]]
            best = minimum(φ(g, a, β, λ) for λ in cands)
            @test v <= best + 1e-9 * (1 + abs(best))
            @test v >= best - 1e-9 * (1 + abs(best))
            @test v <= sum(abs, g) + 1e-12
        end
    end

    @testset "Sampled points stay inside the bounds of a child box" begin
        input_dim = 5
        for trial in 1:6
            layer_dims = [rand(15:40) for _ in 1:rand(3:6)]
            push!(layer_dims, 6)
            N1o, N2o = make_dense_pair(input_dim, layer_dims; relu=true)
            G = GeminiNetwork(N1o, N2o)
            N1 = executable_network(N1o); N2 = executable_network(N2o)

            # Parent: the box [-1, 1]^5. Its bounds cache is inherited by the child.
            VeryDiff.BOUND_ROWS[] = 0
            P0 = PropState(true)
            prepare_prop_state!(P0, create_verification_task(fill(-1.0, input_dim), fill(1.0, input_dim)))
            propagate!(G, P0)

            # Child: the box [-1, 0] × [-1, 1]^4 with the parent's cache and bound rows on (all gaps, many rows).
            lo = fill(-1.0, input_dim); hi = fill(1.0, input_dim); hi[1] = 0.0
            mid = (lo .+ hi) ./ 2
            task = VerificationTask(mid, mid .- lo, collect(1:input_dim), nothing, nothing, nothing, nothing,
                                    nothing, 1.0, 1.0, deepcopy(P0.task_bounds))
            VeryDiff.BOUND_ROWS[] = 1000
            VeryDiff.BOUND_ROWS_RTOL[] = 0.0
            P = PropState(true)
            prepare_prop_state!(P, task)
            propagate!(G, P)
            VeryDiff.BOUND_ROWS[] = 0
            VeryDiff.BOUND_ROWS_RTOL[] = 0.05

            # The test is only useful if the child had rows: some cached bound is tighter than its zonotope bound.
            rows = BoundRow[]
            for L in G.diff_layers
                L.layer1 isa ONNXRelu || continue
                Zi = P.zono_storage.zonotopes[L.inputs[1]].zonotope
                collect_bound_rows!(rows, P.task_bounds.bounds_cache[L.layer_idx], Zi, 1000, 0.0)
            end
            trial == 1 && @info "bound rows in the child box: $(length(rows))"

            Zout = P.zono_storage.zonotopes[end].zonotope
            b1 = zono_bounds(Zout.Z₁); b2 = zono_bounds(Zout.Z₂); bd = zono_bounds(Zout.∂Z)
            bad = 0
            for _ in 1:5000
                x = lo .+ rand(input_dim) .* (hi .- lo)
                y1 = N1(x); y2 = N2(x)
                ok = all(b1[:, 1] .- 1e-8 .<= y1 .<= b1[:, 2] .+ 1e-8) &&
                     all(b2[:, 1] .- 1e-8 .<= y2 .<= b2[:, 2] .+ 1e-8) &&
                     all(bd[:, 1] .- 1e-8 .<= y1 .- y2 .<= bd[:, 2] .+ 1e-8)
                ok || (bad += 1)
            end
            @test bad == 0
        end
    end
end
