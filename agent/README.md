# VeryDiff-Agent (MVP)

An LLM agent drives VeryDiff's differential proof search through MCP tools — **soundly**.
The agent can only take sound proof-tree actions inside a trusted kernel (LCF style); the result
is a certificate that an independent checker replays. Exploration (sampling, local VeryDiff runs,
weight analysis, LPs) happens in a network-less Docker sandbox and never counts as evidence.

MVP scope: **ε-equivalence** (max_i |N1(x)_i − N2(x)_i| ≤ ε on a VNNLIB input box). VeryDiff's
`src/` is used as-is; all code lives in `agent/`.

```
Claude Code / headless harness
        │  MCP (stdio)
        ▼
agent/server/verydiff_mcp  (Python, mcp SDK 2.x MCPServer)
   ├── kernel_client.py ── NDJSON-RPC over stdio ──► agent/julia/VeryDiffAgentKernel (Julia; holds ALL proof state)
   └── sandbox.py ── docker exec ──► persistent container (--network none, read-only rootfs, non-root)
agent/checker/check_certificate.jl   (separate process; does not use the kernel's tree/region/closure code)
```

## Layout

| Path | Contents |
|---|---|
| `julia/VeryDiffAgentKernel/src/` | `Regions.jl` (exact boxes, outward task conversion), `Props.jl` (property adapters), `ProofTree.jl`, `Session.jl` (spec hash, derived closure), `Kernel.jl` (seeded copy of `propagate!`), `Actions.jl`, `AutoExpand.jl`, `Lemmas.jl`, `Inspect.jl` (read-only tools, `read_weights`, exports), `Eval.jl`, `Certificate.jl`, `RPC.jl` |
| `julia/VeryDiffAgentKernel/test/runtests.jl` | unit, differential, lemma, cache-hygiene, checker-mutation and RPC tests |
| `checker/check_certificate.jl` | independent replay checker |
| `server/verydiff_mcp/` | MCP server, kernel client, sandbox, exports |
| `sandbox/` | `Dockerfile`, `build.sh` |
| `harness/` | `run_benchmark.py`, `prompts/agent_system.md` |
| `tests/` | `gen_tiny_nets.py` (+ generated `nets/`), `test_e2e.py`, `test_soundness.py` |

## Setup

Julia 1.11 (VeryDiff's Manifest is resolved with 1.11.8), Python ≥ 3.11, Docker (optional, sandbox only).

```bash
# 1. VeryDiff itself (repo root). Gurobi.jl must be *built* even when Gurobi is not used:
export VERYDIFF_NO_GUROBI=1
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build("Gurobi"); Pkg.precompile()'
# 2. Kernel package (uses this checkout of VeryDiff via Pkg.develop; Manifest is committed)
julia --project=agent/julia/VeryDiffAgentKernel -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# 3. Python
pip install "mcp>=2" numpy onnx onnxruntime claude-agent-sdk
# 4. Sandbox image (optional; stages the host Julia depot so the build is offline)
agent/sandbox/build.sh            # -> verydiff-agent-sandbox:latest
# 5. Optional sysimage to reduce start-up latency
julia --project=agent/julia/VeryDiffAgentKernel agent/julia/VeryDiffAgentKernel/build_sysimage.jl
export VDA_SYSIMAGE=agent/julia/VeryDiffAgentKernel/vdagent_sysimage.so
```

Environment notes from building this in a restricted cloud container: `pkg.julialang.org` may be
blocked by a proxy (`JULIA_PKG_SERVER=""` + General registry from GitHub works), the Gurobi_jll
artifact comes from `packages.gurobi.com`, and `dockerd` may need to be started by hand.

## Running

* **Claude Code:** copy `.mcp.json.example` to your project's `.mcp.json` and fix the paths.
* **Server directly:** `PYTHONPATH=agent/server python3 -m verydiff_mcp.server` (stdio).
  Env: `VDA_WORKDIR`, `VDA_SANDBOX=0|1`, `VDA_SANDBOX_IMAGE`, `VDA_SYSIMAGE`, `VDA_CALL_TIMEOUT`,
  `VDA_WARMUP`, `VDA_FINAL_CERT` (certificate written on exit), `VDA_MAX_RESPONSE`.
* **Checker:** `julia --project=agent/julia/VeryDiffAgentKernel agent/checker/check_certificate.jl CERT NET1 NET2 SPEC`
  → `ACCEPT SAFE|UNSAFE|PARTIAL …` or `REJECT <reason>` (exit 0/1). Certificates may be `.json.gz`.
* **Benchmark:** `python3 agent/harness/run_benchmark.py --agent claude|scripted|none [--baseline] [--timeout S]`
  writes per-instance `calls.jsonl`, `certificate.json` and `summary.csv`.

Latency (this container, 4 vCPU): kernel start + warm-up ≈ 60–75 s before the first tool call
(JIT; a sysimage should remove most of it); afterwards a `check` on the tiny nets takes ~1–20 ms.

### Tools

Proof actions: `open_session`, `check`, `split` (cut / k-way `cuts` / `dim_block`), `auto_expand`,
`introduce_lemma`, `prove_lemma`, `retract_lemma`, `refute_lemma`, `prune`, `submit_counterexample`,
`export_certificate`.
Inspection (read-only): `session_status`, `list_layers`, `list_open_leaves`, `get_node`,
`inspect_layer`, `output_bounds`, `split_scores`, `cache_bounds`, `eval_point`, `read_weights`.
Sandbox: `sandbox_exec`, `sandbox_write_file`, `export_to_sandbox` (tree | regions | bounds | generators,
JSON + NPZ), `sandbox_reset`.

## Kernel design (short)

* **Regions** are exact `Float64` boxes; a split at `t` yields `[lo,t]` and `[t,hi]` with the identical
  `t`. Conversion to VeryDiff's centre/radius widens by one ulp and rounds the radius up; coverage is
  asserted with exact rational arithmetic.
* **Fresh `PropState` per propagation**, initialised from the node's own task. This keeps generator
  allocation consistent (VeryDiff sizes ReLU generator blocks from the *first* task's fresh bounds)
  and makes nodes independent.
* **Bounds caches:** every node owns deep copies — `cache_in` (inherited, frozen) and `cache_out` (after
  its own check). Children inherit `cache_out` if present, else `cache_in`. Checks always start from
  `cache_in`, so they are deterministic and replayable (the certificate's `p` flag records which cache
  the children inherited).
* **Lemmas** (∂ / N1 / N2 bounds at a ReLU input) seed the cache before that ReLU is propagated
  (an uninitialised cache is first set to ±∞ and marked initialised, otherwise VeryDiff overwrites the
  seed). A lemma is its own obligation over the scope node's region, proven by prefix propagation
  (`LemmaBoundsAdapter`). **Lemma chaining:** a lemma's proof tree inherits the lemmas in scope at its
  assume node and that node's bounds cache, so bounds can be proven layer by layer. This is non-circular:
  inherited lemmas were introduced earlier and their proofs cannot mention the new lemma. A lemma counts as
  proven only if its tree is closed *and* every inherited lemma is proven (kernel and checker; the checker
  also detects cycles). Retracting a lemma deletes every lemma introduced below its assume node.
* **Closure is derived**: closed leaf, or all children closed and (for `assume`) the lemma proven.
  SAFE ⟺ main root closed. There is no close/drop action. UNSAFE needs a concretely re-evaluated cex.
* **Guards:** a node is never closed if a cache becomes inconsistent (lower > upper, e.g. from a false
  lemma) or if NaN appears in bounds (see findings below).

## Cost metrics

`session_status`, certificates (`stats`) and `harness/summary.csv` report **nodes** (proof-tree size),
**checks** (VeryDiff propagations done by proof actions: every check, including failed ones, lemma proofs
and work later pruned) and **inspection propagations** (read-only tools). The `min-cost` harness objective
asks the agent to minimise both nodes and checks (`min-nodes` is kept as an alias).

## Trust base

* VeryDiff's transformers (`propagate_layer!`, `init_layer!`), `get_epsilon_property`, `zono_bounds`.
* `Float64` arithmetic **without outward rounding** inside the transformers; the one-ulp widening only
  closes the gap between the exact box and VeryDiff's centre/radius form.
* VNNLib.jl ONNX and VNNLIB parsing; `compute_all_outputs` for concrete evaluation.
* The checker (≈400 lines, re-implements boxes, outward conversion, seeding, closure).
* Spec binding: `spec_hash = sha256(net1, net2, spec, canonical property JSON, git tree hash of VeryDiff's src/)`
  (length-prefixed parts; `-dirty` suffix if `src/` has local changes).

## Findings about VeryDiff (documented, `src/` untouched)

1. **NaN ⇒ "SAFE" in the ε property.** `get_epsilon_property` tests `distance_bound > epsilon`; a NaN
   bound fails that test and the property reports success. NaN can arise in the ReLU transformer when a
   cached interval collapses (`lower == upper`) for a neuron that the *fresh* bounds classify as unstable
   (`α = -l/(u-l)`). The kernel and the checker refuse to close such nodes; plain VeryDiff does not guard.
2. **Selectors from fresh bounds, slopes from cached bounds — changed in `src/` (separate commit
   "ReLU diff transformer: compute phase cases from cached bounds").** Previously `get_selectors` used the
   fresh bounds while α/μ/∂λ used the cached ones, so a unit proven stable by the cache (inherited bound or
   seeded lemma) was still relaxed in ∂. Both now use the cached bounds; cached ⊆ fresh, so cases only move
   from unstable to stable and `init_layer!`'s allocation stays an upper bound. VeryDiff's test suite, the
   kernel tests and the soundness fuzzer pass. Effect (ACAS prop_16, phase-lemma chain 3–18): root bound
   1072 → 4.4, main tree 963 → 515 nodes. Plain VeryDiff runs are unchanged (1483 zonotopes). Note: the
   spec hash includes the `src/` tree hash, so certificates made before this commit no longer verify.
3. **Generator allocation from the first task.** In VeryDiff a `PropState` is reused for all tasks and
   ReLU generator blocks are sized from the root's fresh bounds; a later task with more unstable neurons
   would trip an assertion. The kernel avoids this with a fresh `PropState` per propagation.
4. `split_zono` hands the cache to child 1 by reference (child 2 gets a copy); the kernel never shares.
5. Verified ε semantics refer to the network output **before** a removed final Softmax, while
   `get_epsilon_property` samples counterexample distances with the Softmax still applied. The kernel
   uses pre-Softmax outputs consistently (`softmax_removed` is recorded in the certificate).

## Test status

| Test | Result |
|---|---|
| `julia --project=agent/julia/VeryDiffAgentKernel agent/julia/VeryDiffAgentKernel/test/runtests.jl` | pass (regions, seeding incl. uninitialised cache, derived closure, differential vs CLI on all 5 instances, k-way/partial, UNSAFE, lemmas incl. refutation/retraction, cache hygiene, 8 checker mutations, extension stubs, RPC) |
| `python3 agent/tests/test_soundness.py` | 0 violations in ≈19.7 M sampled points over 4 SAFE certificates (one with a proven lemma) |
| `python3 agent/tests/test_e2e.py [--sandbox]` | MCP → kernel → certificate → checker ACCEPT for SAFE and UNSAFE; sandbox has no network, `/exports` read-only, VeryDiff CLI runs inside |

Differential node counts match VeryDiff's CLI exactly (e.g. `safe_hard`: 3763 nodes in both).

## Extension hooks and what is missing (M8)

* `PropertyAdapter` interface: `adapter_check`, `adapter_split_heuristic`, `needs_lp`, `split_blocks`,
  `stop_layer`, `check_tag`, `violates`. `Top1Adapter` and `RobustnessAdapter` exist as stubs that throw
  `NotImplemented`; `make_property` maps `top1`/`robustness` to that error.
* `dim_block ∈ {primary, sec1, sec2}` is plumbed through the MCP tool, the kernel (`split_blocks` gate,
  stored in the node), and the certificate (`"b"`). The checker currently rejects non-primary blocks.
* `verification_status` is stored per node and passed to the property (Top-1 needs it).
* Missing for Top-1 / δ-Top-1 / robustness:
  - LP inside the kernel (`needs_lp`): Gurobi/GLPK model construction per check, deterministic
    solver settings, and a story for the LP's numerical trust (solver tolerances are not sound).
  - Secondary split blocks: `Regions` needs boxes over [primary; sec1; sec2], `to_task` must fill
    `distance{1,2}_secondary`/`middle{1,2}_secondary`, and the checker must reconstruct them.
  - Concrete semantics (`violates`) for Top-1/δ-Top-1 and the robustness pair (x, x′).
  - VeryDiff's Top-1 property uses a global `FIRST_ROUND`/`TOP1_FOUND_CONCRETE_DELTA` state that must be
    reset per check for replayability.
* Other non-goals still open: halfspace / ReLU-phase splits, arbitrary generator splits, outward-rounded
  arithmetic, multithreading.
