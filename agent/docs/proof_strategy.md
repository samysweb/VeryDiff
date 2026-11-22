# How to build these proofs: lessons and a playbook

These notes come from building equivalence proofs for ACAS prop_16 (1483 → 1229 propagations) and MNIST
mnist_68 (118,707 → 166; 118 with noise-box contraction, since removed), first by letting Claude drive the kernel through MCP, then by heuristic search, and
finally by reading the weights. Detailed numbers are in the experiments branch (`agent/experiments/README.md`).

## What did not work

* **Plain search improvements.** Better split choices for VeryDiff's input splitting (lookahead with dry runs,
  off-centre cuts, a calibrated cost model) gave −9 … −27 %. Useful, but it does not change the shape of the
  problem: the number of boxes grows with the input dimension and the precision needed.
* **Lemmas that only restate what VeryDiff already knows.** Interval/phase lemmas on single units (ACAS, MNIST)
  cost about as much to prove as they saved. Lemmas only pay when they carry information the abstraction lacks.
* **Generic ReLU phase splits** chosen by output-bound attribution (`relu_scores`): each split removes one
  unit's relaxation but the branches still need nearly all of the parent's input splitting; on ACAS 40–48 of 50
  last-layer units were still unstable after many input splits. Two VeryDiff effects made it worse until fixed:
  an "active" branch is the identity over the whole zonotope (the constraint only tightens the caches), and tightening a unit's
  cached bounds could switch the differential transformer to a looser one-sided case (fixed by the
  `ONE_SIDED_DIFF_FALLBACK` flag).
* **Oblique cuts, layer-1 region enumeration, very narrow bands** in the hard MNIST band: all more expensive
  than VeryDiff's axis-aligned splitting there (see `mnist_proof.md` and the experiments branch).

## What worked

1. **Read the program diff first.** For pruning/fine-tuning pairs the difference flows through very few
   channels. On MNIST, one hidden activation (r = h₄₅) explained 94 % of the output difference.
2. **Case-split on the driving hidden quantity, not on inputs.** Bands of r are oblique cuts in input space
   that VeryDiff cannot express; four of ten bands closed with a single propagation each.
3. **Make the constraint count everywhere below the split:** dual-tightened intermediate bounds and the split
   rows in the final LP. Without the dual-tightened bounds the bands did not help (×21 propagations with LP
   rows). Bound rows (cached bounds as extra rows, option `bound_rows`) gave nothing measurable. Noise-box
   contraction was removed: its only benefit came through VeryDiff's split heuristic (`mnist_proof.md`, ablation).
4. **Tighten the final check where the abstraction loses correlation:** the LP-relaxation rows (triangles,
   identities, differential chords) took the hard band from 6905 to 13 propagations.
5. **Measure the true margin early.** Optimising the real networks per region (cheap, numpy/scipy) tells you
   whether a region is hard because the property is tight (margin 0.06–0.13 in MNIST's upper bands) or because
   the abstraction is loose. It also falsifies properties before any proof effort (over the whole MNIST box the
   same pair differs by 40).

## Playbook for the next instance

1. **Falsify and size the margin.** Sample + local optimisation of max |N1 − N2| over the box; if the maximum
   is above ε, submit the counterexample (`submit_counterexample`) and stop.
2. **Read the diff:** `network_diff_structure` (which units/inputs differ, knocked-out units, noise level),
   `read_weights` (with `which=diff`, rows/cols) for the details.
3. **Find the drivers:** exact ranges of the first layer (`inspect_layer` at the first ReLU layer is exact, or
   an LP over the box), `cache_bounds` at deeper layers, and `difference_attribution` on the critical output row
   (which units carry the difference, which lose most when maximised independently). `eval_point` with
   `hidden=True` and `export_to_sandbox` for anything that needs numpy.
4. **Choose the case distinction** on the driving quantity: `split_relu` with thresholds on z1/z2/d at a ReLU
   input (bands), `split_lin` for a linear combination, `split` for input dimensions. Start coarse, check the
   band roots (`check`), refine only bands that do not close. Unchecked split nodes cost no propagation.
5. **Turn on the precision you need:** `open_session(..., fallback=True)` always with hidden splits;
   `lp_relax=True` when the margin is small (each check then costs ~15× a plain propagation, but trees shrink by
   orders of magnitude).
6. **Discharge with VeryDiff:** `auto_expand` on each band (`per_leaf_max_nodes` / `batch` for many leaves),
   watching `session_status`. Use `trial_actions` (dry run, not counted) to compare candidate splits on a
   subtree before committing to one.
7. **Lemmas only for information the abstraction lacks** (e.g. a bound that holds over a large region and is
   reused in many branches); dry-run them with `trial_lemma` first.
8. **Export and check:** `export_certificate`, then the independent checker.

## Tool use by purpose

| Purpose | Tools |
|---|---|
| Program reading | `network_diff_structure`, `read_weights`, `list_layers` |
| Bounds and phases | `inspect_layer`, `cache_bounds`, `output_bounds`, `relu_scores` |
| Concrete behaviour | `eval_point`, `difference_attribution`, sandbox (`export_to_sandbox`, `sandbox_exec`) |
| Case distinctions | `split_relu` (phase or threshold), `split_lin` (linear combination), `split` (input) |
| Discharge | `check`, `auto_expand`, `batch` |
| Planning (not proof cost) | `trial_actions`, `trial_lemma`, `split_scores` |
| Lemmas | `introduce_lemma`, `introduce_phase_lemma`, `introduce_relational_lemma`, `prove_lemma`, `retract_lemma`, `refute_lemma` |
| Bookkeeping | `session_status`, `list_open_leaves`, `get_node`, `prune`, `export_certificate` |

## Cost: propagations versus time

Propagation counts are a good proxy for certificate size and replay cost, but they hide per-node work: with
LP rows one check costs ~0.03–0.05 s on MNIST against ~2.5 ms for a plain propagation, and an LP with
per-unit bound tightening (~600 LPs per node) would cost ~30 s. Report both. For proofs over much larger input
spaces (e.g. all 784 pixels free) input splitting is hopeless, so the expensive per-node primitives
(LP rows, LP-tightened bounds, Farkas refutation of infeasible phase combinations, reusable lemmas) become the
right trade-off.
