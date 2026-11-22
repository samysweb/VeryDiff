# Direct certificate construction (no LLM in the loop)

`lookahead.jl` drives the kernel: at each open node it dry-runs candidate expansions with `trial_actions!`
(each finished by VeryDiff's best-first search, capped at the best cost so far) and applies the cheapest;
subtrees whose best cost is ≤ THR (12) are finished by VeryDiff directly. Cost = checks in the certificate.

ACAS prop_16 (ε = 0.05), plain VeryDiff: 1483 checks.
* input splits, top-3 dims at midpoint + top-4 ReLU units (z1/z2/phase4/d): 1345 checks (−9.3 %), no ReLU split
  ever chosen; exploration 273,658 trial propagations, 546 s.
* `CUTS=1 NRELU=0 NIN=5`: all 5 dims at 1/4, 1/2, 3/4: **1229 checks (−17.1 %)**, checker ACCEPT
  (`acas_prop16_lookahead_cuts_cert.json.gz`); exploration 215,048 trial propagations, 406 s.

Why ReLU splits lose (`diag.jl`, `diag_mnist.jl`):
* ACAS: at depth 6–8, 40–48 of the 50 layer-18 units are still unstable; fixing one phase leaves each branch
  with nearly the parent's input-split work (phase4 ≈ 1.1–3× the input split's cost).
* VeryDiff's differential ReLU transformer is not monotone in the cached bounds. MNIST mnist_68 root, layer-6
  unit 35: z1 ∈ [−1.14, 0.048], z2 ∈ [−0.97, 0.146], d ∈ [−0.259, −0.012] (any_any, relaxed via the narrow d
  interval). Fixing z1 ≤ 0 makes it neg_any, ∂ = −ReLU(z2), relaxed via the wide z2 interval: the child's
  output bound is 1.206 > parent 1.187.
* Active branch: a unit with lower = 0 is treated as the identity over the whole zonotope (including z < 0);
  box contraction of the noise symbols from the constraint (implemented) barely tightens, since one constraint
  spans hundreds of symbols.

## One-sided differential fallback (`src/` commit 549caf4d, flag `ONE_SIDED_DIFF_FALLBACK`, default off)
A unit whose fresh bounds give any_any but whose tighter cached bounds give a one-sided case keeps the d-based
relaxation (sound: min(0,d) ≤ ReLU(z1) − ReLU(z2) ≤ max(0,d)). Kernel option `fallback` (certificate option,
checker sets the flag). Sampled soundness (`fallback_sound.jl`): 63,720 tensor samples (all layers, z1/z2/d) on the
tiny nets, ACAS and MNIST nodes, 0 violations. Plain VeryDiff with the flag: ACAS 1484, MNIST 118,708 checks
(unchanged). An earlier variant (switch whenever the d interval is narrower) made MNIST worse (129,978).
MNIST root, single ReLU splits with the flag: no child is looser than the parent (1.187); before up to 1.31.
Both-active branches of a 4-way phase split are still looser (pos_pos passes d unchanged; not covered).

## MNIST mnist_68_local_15 (ε = 1.0), plain VeryDiff: 118,707 checks
`proxy_search.jl`: best-first; at each open node try the top-NIN input dims (and the top-NRELU relu_scores units,
z1/z2 single splits), check the children, keep the action with the smallest estimated subtree cost
Σ f(excess_child). f = subtree size vs bound excess, calibrated on VeryDiff's own tree for this instance
(`calib.jl`, `costtab.tsv`; ~5 nodes at excess 0.01, ~97 at 0.1, ~2000 at 0.2). The calibration uses a full
VeryDiff run of the same instance: this is a certificate-size experiment, not a faster verifier.
* input only (NIN=2, NRELU=0): **86,901 checks (−26.8 %)**, checker ACCEPT (`mnist68_proxy_input_cert.json.gz`);
  exploration 173,800 trial checks, 815 s (+ the 118,707-check calibration run).
* with ReLU splits (NRELU=4, fallback on): stopped at 72,001 nodes with 28,205 open leaves (input-only had
  finished at 86,901 in total) — clearly worse. 12,476 of 36,000 expansions were ReLU splits; the proxy accepts
  them when a branch's bound drops, but the branches keep needing input splits (VeryDiff's input heuristic does
  not see the constraint), so the tree gets wider.
