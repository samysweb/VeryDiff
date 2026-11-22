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
