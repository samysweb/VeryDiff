> Files referenced below were moved into numbered experiment directories; see `README.md` for the map.

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

## MNIST by reading the weights (`mnist68_proof.jl`): 24,802 propagations vs 118,707 (−79 %), checker ACCEPT
Facts from the weights (`network_diff_structure`, `load.py`/`sample.py`, `difference_attribution`):
N2 = N1 with layer-1 units {31,38,45,80,95} and layer-2 units {10,37,38,51,86} knocked out; other weights equal to
4.2e-4. Over the box, pruned layer-1 units 31/38/80/95 are always off in N1, unit 45 always on with
r = h45(x) ∈ [0.058, 1.133] (exact: affine in the 15 free pixels). The critical output 4 is a stable identity
output, diff_4 ≈ 0.2 + 0.65 r (sampled max 0.923). Root slack: sum of per-unit maxima of the 21 mixed layer-2
units is 1.10 (vs true 0.92), VeryDiff 1.187.
Proof: case split on r (z1 of unit 45 at ReLU layer 3) into bands; inside each band the constraint is used by
three kernel mechanisms (all re-checked by the independent checker): noise-box contraction, dual-tightened
intermediate bounds at every later ReLU layer (closed-form single-row weak duality), and LP rows in the final
check. Per band (propagations): r ≤ 0.55: 1, [0.55,0.65]: 1, [0.65,0.75]: 1, [0.75,0.85]: 1, [0.85,0.95]: 1857,
[0.95,0.975]: 3827, [0.975,1.0]: 6905, [1.0,1.025]: 6789, [1.025,1.05]: 3449, [1.05,1.133]: 1965.
Development cost: one failed band layout ([0.95,1.05] did not close within 30k checks), a few hundred probe
propagations, sampling in numpy. Phase splits on the high-loss units (N2 unit 69, 34, 50, 76) only gained
0.02–0.08 in one branch and were not used.

## Hard subgoal r ∈ [0.95, 1.05] (`hard_band/`): no better strategy found
Current proof: four 0.025-wide bands, 3827 + 6905 + 6789 + 3449 = 20,970 propagations (a single band did not close
within 30k). Findings:
* True maxima (optimised, `truemax.py`): [0.95,0.975] 0.882, [0.975,1.0] 0.895, [1.0,1.025] 0.901, [1.025,1.05]
  0.906 — margin ≈ 0.1. Zonotope bound in [0.975,1.0]: 1.109 = centre 0.601 + input symbols 0.282 + layer-2
  differential relaxations 0.211 + rest; the LP with the band rows gives 1.064.
* In a narrow band Δ_j = W[j,45]·r is almost constant, so each mixed layer-2 unit contributes a step
  clamp(h_j, 0, Δ_j); no convex single-unit relaxation captures it, and ~12 such independent steps add up.
  Inside the slab the true difference is nearly linear (residual std 0.008–0.019), but 35 units have phase
  boundaries through it: 376 distinct activation patterns in 680 samples (not saturated).
* Tried, all worse than the 0.025 bands: layer-1 region enumeration by phase splits (6801 / 30,860 vs 3827 / 6905;
  no branch refuted, since single-row duality cannot detect joint infeasibility); one oblique cut along the
  in-band linear model (new primitive `split_lin`; 11,478 vs 6530); 4-way phase splits on the top layer-2 units
  (one branch infeasible, the others 1.01–1.11 vs 1.064); 0.0125-wide bands (27,120 vs 20,970).

## LP-relaxation rows (`lp_relax`): MNIST in 124 propagations (checker ACCEPT)
Kernel option `lp_relax`: the final check adds rows that are true relations at every unstable unit, evaluated on
the zonotope rows of the shared witness ε: per-network triangles (a ≥ 0, a ≥ z, chord), identities d = z1 − z2
and t = a1 − a2, and differential chords min(0,d) ≤ t ≤ max(0,d). The certificate lists the rows (layer, unit,
type); the checker rebuilds them from its own replay (formulas valid for any sound bounds) and re-verifies λ.
GLPK gets each identity pair as a band |row| ≤ 1e-7·scale (exact equalities are numerically overdetermined:
138 identities over 111 symbols; two almost coincident half-spaces made GLPK report infeasibility), and has a 2 s
time limit (it stalled on a test net); λ is always re-evaluated against the certificate rows.
* Band [0.975,1.0]: bound 1.021 (before 1.064; offline full LP 1.011), closes with 13 propagations / 3.5 s
  (before 6905 / 49 s). Whole proof with the same 10 bands: **124 propagations, 127 nodes, 36 s wall**,
  checker ACCEPT (51 s) — `mnist68_lp_relax_cert.json.gz`. Coarser layouts: 5 cuts 134, 3 cuts 209,
  1 cut 1314 propagations. Without bands (input splitting + LP only) the run did not finish within 20 min.
* Cost per check with the LP: ~0.03–0.05 s on MNIST (vs ~2.5 ms for a plain propagation).
* Tests: witness test (a single ε reproducing all layers at sampled points satisfies every relaxation row),
  complete proof + checker ACCEPT, tampered row type / unit / option / multipliers rejected.
