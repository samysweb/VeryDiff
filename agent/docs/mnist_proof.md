# The MNIST proof: `mnist_3_100` vs `mnist_3_100_pruned5`, `mnist_68_local_15`, ε = 1.0

Property: for every input x in the VNNLIB box, max_i |N1(x)_i − N2(x)_i| ≤ 1 (ε-equivalence, outputs before
the removed Softmax). The box fixes 769 of the 784 pixels; 15 pixels are free in [0, 1]. Instance files come from
verydiff-experiments (`mnist-prune`); the instance description is in the experiments branch
(`agent/experiments/instances/mnist68/`).

## Variants

All numbers are from this container (4 vCPU). "Propagations" = VeryDiff propagations a replay of the
certificate performs (one per checked node). "Split decisions" = internal split nodes of the certificate.
Search time = kernel time to find the proof (excluding Julia start-up, ≈ 60 s); check time = the
independent checker on the certificate (including start-up).

| Variant | How | Propagations | Split decisions | Search time | Check time |
|---|---|---|---|---|---|
| VeryDiff CLI | `Cli.jl`, default heuristic | 118,707 | 59,353 input | 193 s | – (no certificate) |
| Kernel, plain | `auto_expand` from the root (same tree as the CLI) | 118,708 | 59,353 input | 297 s | not re-measured (22 MB certificate, ACCEPT) |
| Heuristic search | one-step proxy over input splits (`proxy_search.jl`) | 86,901 | 43,450 input | 815 s (+ one full VeryDiff run for calibration, 174k trial propagations) | 204 s |
| Weight reasoning | 10 bands on r = h₄₅ (`mnist68_proof.jl`, `LP_RELAX=0`) | 24,796 | 9 band + 12,393 input | 218 s | 192 s |
| **Weight reasoning + LP rows** | same bands, `lp_relax=true` (`mnist68_proof.jl`) | **118** | **9 band + 54 input** | **38 s** | **51 s** |

(The Opus-via-MCP runs on this instance reached 118,707 and 118,691 nodes; see the experiments branch.)
The last variant's search performs 124 checks because the script checks every band root once before handing
it to `auto_expand`, which checks open roots again; the certificate needs 118.

Coarser band layouts with LP rows: cuts {0.65, 0.85, 0.95, 1.0, 1.05} → 134 propagations; {0.85, 0.95, 1.05}
→ 209; {0.75, 0.9, 1.0} → 268; {0.9} → 1314. Without any band (input splitting + LP rows only) the search had not
finished after 20 minutes.

## How the short proof works

### 1. Reading the program

`network_diff_structure` (and `read_weights`) show that N2 is N1 with ten units knocked out: layer-1 units
{31, 38, 45, 80, 95} and layer-2 units {10, 37, 38, 51, 86} (1-based) have zero weights and bias −1 in N2, so
they are off everywhere; all other weights agree to 4.2·10⁻⁴ (quantisation noise).

Layer-1 pre-activations are affine in the 15 free pixels, so their exact range over the box is a small LP (or
interval arithmetic). In N1 the pruned units 31, 38, 80, 95 are always off; unit 45 is always on with

  r := h₄₅(x) ∈ [0.058, 1.133].

So after layer 1 the networks differ only in r, and the layer-2 pre-activation difference is W₂[:, 45]·r
(plus noise). The last layer is the identity on ReLU(h₃); the critical output is 4 (h₃,₄ ∈ [10.8, 13.5], always
on), so its difference is linear in the layer-2 activations. Sampling (`difference_attribution`, numpy) gives
diff₄ ≈ 0.2 + 0.65 r (correlation 0.97); the maximum, found by local optimisation, is ≈ 0.937 at large r.

VeryDiff's root bound is 1.187. The attribution shows why: 21 layer-2 units change phase over the box; adding
their terms' maxima independently already gives 1.10 — the zonotope cannot see that these maxima occur at
different inputs.

### 2. The case distinction

The one quantity that drives the difference is r, a single hidden pre-activation. The proof therefore splits
on it (`split_relu` on z1 of unit 45 at the input of ReLU layer 3, with thresholds instead of 0) into 10 bands:

  r ≤ 0.55, [0.55, 0.65], [0.65, 0.75], [0.75, 0.85], [0.85, 0.95], [0.95, 0.975], [0.975, 1.0], [1.0, 1.025],
  [1.025, 1.05], r ≥ 1.05.

The split nodes are not checked (they cost no propagation). Inside a band the constraint lo ≤ r ≤ hi is used by
four kernel mechanisms (all re-checked by the independent checker):

* **Contraction** of the shared noise box and rescaling of all live zonotopes (weak here: one constraint over
  hundreds of symbols).
* **Dual-tightened intermediate bounds:** at every later ReLU layer each unit's z1 / z2 / d bound is tightened by
  weak duality with the band rows (closed form, no LP). This turns "r is in a narrow band" into narrow
  d-intervals for all layer-2 units, so VeryDiff's differential relaxation becomes tight.
* **LP rows for the split constraints** in the final check (weak-duality certificate, multipliers proposed by
  GLPK and re-verified).
* **LP-relaxation rows** (`lp_relax`): for every unstable unit, the per-network triangle (a ≥ 0, a ≥ z, chord),
  the identities d = z1 − z2 and t = a1 − a2 at the shared witness, and the differential chords
  min(0, d) ≤ t ≤ max(0, d). This is what VeryDiff's relaxation lacks: it ties each unit's output back to its
  input across both networks, recovering most of the correlation lost by independent relaxation noise.

### 3. What VeryDiff still has to do

| Band of r | Without LP rows | With LP rows (propagations / input splits) |
|---|---|---|
| r ≤ 0.55, [0.55, 0.65], [0.65, 0.75], [0.75, 0.85] | 1 each | 1 / 0 each |
| [0.85, 0.95] | 1857 | 11 / 5 |
| [0.95, 0.975] | 3827 | 9 / 4 |
| [0.975, 1.0] | 6905 | 13 / 6 |
| [1.0, 1.025] | 6789 | 21 / 10 |
| [1.025, 1.05] | 3449 | 15 / 7 |
| r ≥ 1.05 | 1965 | 45 / 22 |

The upper bands are hard because the true
margin is small: optimised maxima of diff₄ per band are 0.871, 0.882, 0.895, 0.901, 0.906 and 0.937. Without the
LP rows each band's bound stays at 1.06–1.13 and VeryDiff needs ~17 levels of input splitting; with them the
bound of [0.975, 1.0] is already 1.021 at the band root and a handful of input splits close it.

### Dual-tightened bounds: the idea

**Problem.** A hidden case split such as "r ≤ 0.85" is a fact about the inputs of the node. VeryDiff does not
know this fact. It calculates the bounds of each hidden unit from the zonotope of the whole input box. Thus a band
node gets the same layer-2 bounds as the full box, the relaxation of each unstable unit stays wide, and the band
does not help. Without dual tightening, the bands did not close.

**Idea.** In one propagation, all quantities are affine functions of the same noise symbols ε ∈ [−1, 1]^m.
The split constraint is one linear row a·ε ≤ β (for example r(ε) ≤ 0.85). A hidden unit is u(ε) = c + g·ε.
For each λ ≥ 0:

  u = c + (g − λa)·ε + λ·(a·ε) ≤ c + ‖g − λa‖₁ + λβ

The first term is at most ‖g − λa‖₁ because |ε_i| ≤ 1. The second term is at most λβ because of the constraint.

* λ = 0 gives VeryDiff's normal bound. A larger λ removes from g the part that has the direction of the
  constraint. The constraint, not the box, limits that part.
* The best λ is easy to find. The right side is convex and piecewise linear in λ, with break points at g_i / a_i.
  Sort the break points and walk through them until the slope is not negative. No LP solver is necessary.
* Lower bounds use the same method with −g.

**Why it works here.** The layer-2 differences are d_j ≈ W[j, 45]·r (+ noise), so g is almost parallel to the
constraint row of r. With the best λ, almost all of g cancels, and the bound of d_j becomes approximately
W[j, 45]·[band] instead of W[j, 45]·[0.058, 1.133]. VeryDiff then relaxes each layer-2 unit over this narrow
interval, and some units become stable.

**Where it runs.** At each ReLU layer at or after the constraint's layer, for z1, z2 and d of every unit. The
result goes into the bounds cache before VeryDiff's transformer runs.

**Soundness.** The true values of each concrete input satisfy the constraint, and the inequality is true for each
λ ≥ 0. Thus a bad λ only gives a looser bound, never a wrong one. The checker calculates λ and the bound again
with its own code, with outward rounding.

**Compared with contraction.** Contraction (the other use of the constraint) makes the interval of each noise
symbol smaller, one symbol at a time. This is weak when the constraint has hundreds of symbols. Dual tightening
uses the full direction of the constraint for each unit, so it is strong when a unit depends on the same
direction as the constraint.

**Limits.** It uses one constraint row at a time: with many splits it takes the best single row, so it cannot
combine rows and cannot show that two constraints together have no solution (Farkas certificates would add
this). It tightens the cached bounds, not the zonotope. The cost is O(m log m) for each unit and bound.

### 4. Soundness notes

* Every row used by the LP is a true relation between concrete values, evaluated on zonotope rows that one
  witness ε reproduces simultaneously (tested: for sampled concrete points a witness with residual < 10⁻⁸
  exists and satisfies every relaxation row).
* The checker never solves an LP: it rebuilds each row from its own replay (formulas valid for any sound
  bounds) and re-evaluates the bound c + ‖g − Aᵀλ‖₁ + λ·r for the certificate's λ ≥ 0.
* The proof uses the `ONE_SIDED_DIFF_FALLBACK` flag (VeryDiff `src/`, default off): when a tighter cache turns a
  fresh any_any unit into a one-sided case, the d-based relaxation is kept (sound because
  min(0, d) ≤ ReLU(z1) − ReLU(z2) ≤ max(0, d)).

## Reproducing

```bash
cd agent/julia/VeryDiffAgentKernel
INST=/path/to/mnist68  OUT=mnist68.json.gz  VERYDIFF_NO_GUROBI=1 julia --project=. ../../docs/examples/mnist68_proof.jl
VERYDIFF_NO_GUROBI=1 julia --project=. ../../checker/check_certificate.jl mnist68.json.gz $INST/net1.onnx $INST/net2.onnx $INST/spec.vnnlib
```

`LP_RELAX=0` reproduces the 24,796-propagation variant. `INST` must contain `net1.onnx`, `net2.onnx`,
`spec.vnnlib` (the experiments branch has them under `agent/experiments/instances/mnist68/`).
