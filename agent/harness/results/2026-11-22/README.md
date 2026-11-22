# Smoke runs, 2026-11-22 (tiny generated instances)

* `scripted_vs_cli.csv`: scripted agent (check root, budgeted auto_expand) vs VeryDiff CLI on all 5
  instances: identical verdicts, all certificates ACCEPTed by the checker.
* `claude_verdict_safe_hard.csv`: Claude via the Agent SDK, default objective. The agent checked the root
  and delegated to auto_expand → tree identical to the naive run (3763 nodes). 7 tool calls, SDK-reported
  cost $0.15.
* `claude_minnodes_safe_hard.csv` (+ `calls.jsonl`): objective "minimise proof nodes", auto_expand capped at
  100 nodes/call. The agent wrote a lookahead planner in the sandbox (rolling out VeryDiff on candidate
  sub-boxes), then applied 11 manual splits (off-centre and k-way cuts) and finished the sub-boxes with
  auto_expand: 3441 nodes (−8.6 %), 1721 leaves (vs 1882), checker ACCEPT. 85 tool calls, ~10.6 min,
  SDK-reported cost $0.91. No lemmas; `read_weights` was not used (weights were read via numpy in the sandbox).
* `tree_comparison_safe_hard.txt`: output of `harness/compare_trees.py` for the three trees.
* `claude_opus_weights_safe_hard.csv` (+ `calls.jsonl` incl. the agent's text): model claude-opus-5-5
  (Claude Code also used claude-haiku-5-5 for background tasks), prompt with the required weight-analysis
  step, batch + per-leaf auto_expand available. The agent read all layer diffs, wrote a weight report (dense,
  full-rank noise perturbation in every layer → no structural lever, lemmas judged not worth it), split at
  x₂=0 and finished with per-leaf auto_expand using the `widest` heuristic: 3503 nodes (−6.9 % vs naive),
  32 tool calls, 2.5 min, SDK-reported cost $0.43. Caveats: (1) the sandbox was down (dockerd had died);
  (2) `read_weights` then reported Linear-layer biases as 0 because these ONNX nets keep the bias in a separate
  AddConst layer, so the agent's "all first-layer hyperplanes pass through the origin" hypothesis was based on
  a tool artefact. Fixed afterwards: `read_weights` now folds the following AddConst bias in (`b_note`).
* ACAS Xu (verydiff-experiments, `acas-prune`): `ACASXU_run2a_1_1_batch_2000.onnx` vs `..._pruned5.onnx`,
  `prop_16.vnnlib`, ε = 0.05. VeryDiff CLI: SAFE, 1483 zonotopes. Scripted kernel run: identical tree (1483),
  checker ACCEPT. Opus (min-nodes, weight analysis, sandbox): SAFE, 1483 nodes, checker ACCEPT, 48 tool calls,
  13 min, SDK-reported cost $1.35. It correctly identified the pruning (2 dead neurons/layer + small noise,
  ≈rank-1 diffs), located the three pruned neurons that change phase (all at low x5), split x5 into bands
  and used batch/per-leaf auto_expand; a layer-9 lemma (sampled bounds + margin) halved the root bound but
  was not cheaply provable, and lemma proofs cannot use lemmas introduced above them, so no lemma chain.
  Its sandbox experiments found no input-split plan below ≈1400 nodes.
* ACAS prop_16 rerun with lemma chaining available (`acas_prop16_opus_chaining*`): SAFE, 1451 nodes (−2.2 % vs
  1483), checker ACCEPT, 43 tool calls, 12.6 min, SDK-reported cost $1.45. Opus planned a chain ReLU 9→12→15→18
  but dropped it after the first link: the ReLU-9 lemma halved the root bound (1537→765) yet its own proof had
  87/94 leaves open after 100 nodes. Final proof: a 6×3×6 input grid chosen from sandbox experiments, finished
  per cell with auto_expand.
* ACAS prop_16, third Opus run (phase lemmas available; prompt now contains the measured chain costs *of this
  instance*, so this run is not an unbiased benchmark): SAFE, 1483 nodes, checker ACCEPT, 45 tool calls, 17 min,
  SDK-reported $1.46. Opus cited the measured example and added no lemmas; it tried to save nodes by merging
  VeryDiff's consecutive same-dimension splits into k-way splits, which cannot help because the kernel records
  a k-way split as a chain of binary splits (identical node count).
* Phase-lemma chain measured directly (no LLM): dead units at ReLU 3–18 as chained phase lemmas → main tree
  963 nodes (vs 1483) but 5682 lemma-proof nodes; chains up to layer 9/12 cost 293/594 proof nodes and leave
  the main tree unchanged (1486/1487). Certificate of the full chain: checker ACCEPT (6645 nodes, 6 lemmas).
* After the `src/` change "ReLU diff transformer: compute phase cases from cached bounds" (direct measurement,
  ACAS prop_16, phase-lemma chains; nodes / checks):
  plain VeryDiff 1483 / 1483 (unchanged); chain 3–12: main 1487 + lemmas 594 = 2081 nodes / 2082 checks;
  chain 3–18: root bound 1537 → 4.4, main 515 + lemmas 5682 = 6197 nodes / 6198 checks (before the change:
  main 963). Checker ACCEPT; soundness fuzz (5.1 M points, incl. phase-lemma bounds): 0 violations.
* ACAS prop_16, Opus, `min-cost` objective, after the `src/` change (sandbox image rebuilt): SAFE, 1483 nodes /
  1484 checks, checker ACCEPT, 25 tool calls, 16.7 min, SDK-reported $0.93. Opus identified the key unit
  (pruned unit 49 of layer 5, off in net1 by a margin of only 0.6): seeding it would cut the main tree to ≈985,
  but proving it was estimated at ≈2405 nodes in the sandbox, so it used plain best-first splitting.
* ACAS prop_16, Opus with `trial_lemma` + localisation recipe: SAFE, 1483 nodes / 1483 checks, checker ACCEPT,
  33 tool calls, 11 min, SDK-reported $0.90. It split at x5 = 0 (hard half 1377, easy half 105) and dry-ran a
  phase lemma (16 dead units of ReLU 18) in the hard half: proof 919 nodes, saving 30 → net −889; no lemmas used.
* Direct `trial_lemma` of the layer-5 unit-49 phase lemma (net1 off; `trial_unit49.jl`), net saving in nodes:
  whole box −1907 (proof 2405, main 1483 → 985); x5<0 half −1761 (2223, 1377 → 915);
  cells x5∈[−0.5,−0.375] −669, [−0.375,−0.25] −747, [−0.25,0] −343. The lemma never pays off, at any placement.
  (The kernel's numbers equal Opus's earlier sandbox proxies, 2405/985.)
* ε sweep, unit-49 phase lemma on the whole box (`eps_sweep_unit49.jl`), nodes:
  ε=0.05: proof 2405, main 1483→985 (net −1907); ε=0.02: 2405, 2209→1435 (−1631);
  ε=0.01: 2405, 2869→2057 (−1593); ε=0.005: 2405, 3875→2985 (−1515). Proof cost is independent of ε; the
  saving grows only slowly (498→890) because the main tree grows roughly logarithmically in 1/ε.
* MNIST (verydiff-experiments `mnist-prune`): `mnist_relu_3_100` vs `_pruned5`, `mnist_68_local_15` (15 free
  pixels), ε = 1.0. Old logs: SAFE in 71k zonotopes (120 s timeout runs); current VeryDiff CLI: SAFE, 118,707
  zonotopes, 193 s. Scripted kernel run: identical tree, 118,707 nodes / 118,708 checks, checker ACCEPT
  (certificate 22 MB). Opus (min-cost, trial_lemma, sandbox): SAFE, 118,707 nodes / 118,767 checks, checker
  ACCEPT, 36 tool calls, 16.6 min, SDK-reported $0.75. Opus: true max |Δ| ≈ 0.937 vs ε = 1 (thin margin);
  per-neuron bounds are already near the sampled ranges, the looseness is lost correlation between neurons at
  the output (width 9.2 vs 3.8 sampled), which per-neuron interval/phase lemmas cannot express; a one-unit phase
  lemma dry run was still open after 465 nodes. No lemmas used.
