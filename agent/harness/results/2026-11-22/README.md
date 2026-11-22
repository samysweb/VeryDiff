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
