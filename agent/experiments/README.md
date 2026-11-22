# VeryDiff-Agent experiments (2026-11-22)

This branch (`agent/mcp-prototype-experiments`) = the code branch `agent/mcp-prototype` + everything
experimental: agent transcripts, run logs, certificates, analysis and driver scripts, and the two benchmark
instances. Code and documentation live on `agent/mcp-prototype`; start with `agent/README.md` and
`agent/docs/` (`mnist_proof.md`, `proof_strategy.md`, `mcp_tools.md`, `certificate_format.md`).

Detailed chronological notes: [`notes_agent_runs.md`](notes_agent_runs.md) (runs with Claude via MCP and
direct lemma measurements) and [`notes_direct.md`](notes_direct.md) (direct certificate construction,
fallback evaluation, weight reasoning, hard band, LP rows). They were written before the files were moved;
the table below maps every experiment to its directory.

## Running the scripts

Julia scripts use the kernel project; instance paths default to `instances/…` and can be overridden with
`INST=<dir>` (a directory with `net1.onnx`, `net2.onnx`, `spec.vnnlib`):

```bash
cd <repo root>
VERYDIFF_NO_GUROBI=1 julia --project=agent/julia/VeryDiffAgentKernel agent/experiments/09_lp_relaxation/relax_band.jl
python3 agent/experiments/09_lp_relaxation/trilp.py 0.975,0.98125        # needs numpy, onnx, scipy
VERYDIFF_NO_GUROBI=1 julia --project=agent/julia/VeryDiffAgentKernel agent/checker/check_certificate.jl \
    agent/experiments/09_lp_relaxation/mnist68_lp_relax_cert.json.gz agent/experiments/instances/mnist68/{net1.onnx,net2.onnx,spec.vnnlib}
```

Certificates are bound to VeryDiff's `src/` tree hash: those made before a `src/` commit (noted below) only
verify with that commit checked out. Logs (`*.log`) are raw console output; `*_calls.jsonl` are full MCP
transcripts of Claude runs (tool calls, results, assistant text); `*.csv` are harness summaries.

## Instances (`instances/`)

| Directory | Source (verydiff-experiments) | Property |
|---|---|---|
| `acas_prop16/` | `acas-prune`: `ACASXU_run2a_1_1_batch_2000.onnx` vs `…_pruned5.onnx`, `prop_16.vnnlib` | ε = 0.05 |
| `mnist68/` | `mnist-prune`: `mnist_relu_3_100` vs `mnist_relu_3_100_pruned5`, `mnist_68_local_15` (15 free pixels) | ε = 1.0 |

The tiny generated instances are produced by `agent/tests/gen_tiny_nets.py`.

## Experiments

| # | Directory | Question | Main result |
|---|---|---|---|
| 01 | `01_tiny_nets_claude/` | Does the kernel reproduce VeryDiff, and what does Claude do with it on tiny nets? | scripted = CLI on all 5 instances; Claude `min-nodes` 3441 vs 3763 nodes (−8.6 %), Opus with weight analysis 3503 |
| 02 | `02_acas_prop16_claude/` | Can Opus find better ACAS proofs via MCP (weights, lemmas, chaining, phase lemmas, `trial_lemma`)? | best 1451 vs 1483 nodes; lemmas never paid off (direct measurements in `trial_unit49.jl`, `eps_sweep_unit49.jl`, `chain*.jl`) |
| 03 | `03_acas_prop16_direct_search/` | How far does a lookahead search over input splits get (no LLM)? | 1345 (midpoint cuts), **1229** (cuts at ¼, ½, ¾), checker ACCEPT; ReLU splits never chosen (`diag.jl`) |
| 04 | `04_mnist68_claude/` | Opus on MNIST via MCP (with and without relational lemmas / LP) | 118,707 and 118,691 vs 118,707 (CLI log included) |
| 05 | `05_mnist68_direct_search/` | Heuristic proof search with a calibrated cost proxy | 86,901 propagations (−26.8 %), checker ACCEPT; with ReLU splits worse (stopped) |
| 06 | `06_mnist68_weight_reasoning/` | Proof from reading the weights: case split on r = h₄₅ | 24,796 propagations (bands, no LP rows), checker ACCEPT |
| 07 | `07_mnist68_hard_band/` | Better strategy for the hard band r ∈ [0.95, 1.05]? | none found without LP rows: layer-1 regions, oblique cut, phase splits, narrower bands all worse; true band maxima 0.88–0.91 |
| 08 | `08_onesided_fallback/` | Is VeryDiff's differential ReLU non-monotone, and is the fix sound? | fixed by `src/` flag `ONE_SIDED_DIFF_FALLBACK` (commit `549caf4d`); 63,720 sampled tensors, 0 violations; plain runs unchanged |
| 09 | `09_lp_relaxation/` | LP-relaxation rows in the final check | MNIST in **118 propagations** (124 checks), 38 s, checker ACCEPT |
| 10 | `10_full_box_falsification/` | Does the MNIST pair satisfy ε = 1 over all of [0,1]^784? | no: max difference ≥ 40.1 |

### 01 `01_tiny_nets_claude/`
`scripted_vs_cli.csv`; `claude_verdict_safe_hard.csv`; `claude_minnodes_safe_hard.csv` + `_calls.jsonl`;
`claude_opus_weights_safe_hard.csv` + `_calls.jsonl`; `tree_comparison_safe_hard.txt`
(`agent/harness/compare_trees.py`).

### 02 `02_acas_prop16_claude/`
Opus runs, each `acas_prop16_opus_<variant>.csv` + `_calls.jsonl` + console log `acas_opus*.log`:
`opus` (weight analysis; 1483), `opus_chaining` (lemma chaining; 1451), `opus_phase` (phase lemmas; 1483),
`opus_mincost` (after the selector `src/` fix; 1483), `opus_trial` (`trial_lemma`; 1483); `acas_prop16_scripted.csv`.
Direct measurements: `chain.jl`, `chain9.jl`, `chain12.jl` (phase-lemma chains), `trial_unit49.jl`,
`eps_sweep_unit49.jl` / `eps_sweep.jl` (ε sweep of the unit-49 lemma).

### 03 `03_acas_prop16_direct_search/`
`lookahead.jl` (dry-run lookahead over candidate expansions; env `THR`, `NIN`, `NRELU`, `CUTS`),
`acas_lookahead.log` (1345), `acas_lookahead_cuts.log` (1229; rerun with the current `src/` tree: `acas_lookahead_cuts_rerun.log`, identical) + `acas_prop16_lookahead_cuts_cert.json.gz` (from the rerun),
`diag.jl` (costs of ReLU splits vs input splits at mid-depth nodes).

### 04 `04_mnist68_claude/`
`mnist68_opus.csv` + `_calls.jsonl` (118,707), `mnist68_opus_lp.csv` + `_calls.jsonl` (relational lemmas / LP,
118,691), `mnist68_scripted.csv`, logs `mnist_opus.log`, `mnist_opus_lp.log`, `mnist_scripted.log`, and the
VeryDiff CLI log `cli_mnist_68_local_15.log` (118,707 zonotopes).

### 05 `05_mnist68_direct_search/`
`calib.jl` (subtree size vs bound excess from VeryDiff's own tree → `costtab.tsv`), `proxy_search.jl`
(best-first with one-step proxy; env `NIN`, `NRELU`, `FALLBACK`, `MAXN`), logs `mnist_proxy_input.log`
(input only: 86,901) and `mnist_proxy.log` (with ReLU splits, stopped at 72k nodes), certificate
`mnist68_proxy_input_cert.json.gz`, `time_step.jl` (per-step timing).

### 06 `06_mnist68_weight_reasoning/`
`load.py` (ONNX → numpy, spec parsing), `sample.py` (sampled per-unit decomposition of diff₄), `instr.jl`
(`network_diff_structure`, `difference_attribution`), `t68.jl` (phase split on N2 unit 69), `tr.jl` and
`bands.jl` (thresholds on r), `probe.jl` (narrow bands × phase splits), `topbands.jl` (bands discharged by
VeryDiff), certificate `mnist68_weight_reasoning_cert.json.gz` (24,796 propagations). The final script is
`agent/docs/examples/mnist68_proof.jl` (`LP_RELAX=0` reproduces this variant).

### 07 `07_mnist68_hard_band/`
`slab.py` (sampling of a band, exact LP ranges), `truemax.py` (optimised true maxima per band),
`band_attr.jl`, `decomp.jl` (bound decomposition by generator source), `leafstats.jl`, `regions.jl`
(layer-1 region enumeration), `oblique.py` + `oblique_run.jl` + `oblique_0.975_1.0.json` (oblique cut),
`p4.jl` (4-way phase splits).

### 08 `08_onesided_fallback/`
`diag_mnist.jl` (child bounds of ReLU splits at the MNIST root; env `FALLBACK=1`), `fallback_sound.jl`
(sampled soundness of all layers with the flag), `fallback_cost.jl` (plain VeryDiff with the flag).

### 09 `09_lp_relaxation/`
`trilp.py` (offline triangle + differential LP with LP-tightened bounds), `relax_band.jl` (kernel bounds per
band), `relax_auto.jl` (one band discharged; env `W` for sub-bands, `RELAX`, `OUT`), `relax_full.jl` (whole proof
for a cut list), `relax_time.jl` (cost on the test nets). Debugging the GLPK infeasibility: `lpdebug*.jl`
(row-group feasibility), `cachecheck.jl` + `band_pts.txt` (cache soundness on band points), `alias.jl`
(buffer aliasing), `witness.jl` / `witness2.jl` (single-witness test). Final run `final.log`, check
`final_check.log`, certificate `mnist68_lp_relax_cert.json.gz` (127 nodes, 118 propagations).

### 10 `10_full_box_falsification/`
`falsify_full_box.py`: random images and sign-gradient ascent over the whole MNIST box.

## `src/` commits that affect certificates

The checker requires the certificate's `verydiff_src_tree` to equal the local `src/` tree. All certificates on
this branch were made with the current tree (`3381bc19…`, i.e. after `549caf4d`, `ONE_SIDED_DIFF_FALLBACK`).
For the Claude runs in 01, 02 and 04 only summaries and transcripts were kept (their certificates were checked
at the time, partly against the tree before `d9979d3`, phase cases from cached bounds).
