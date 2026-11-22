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
