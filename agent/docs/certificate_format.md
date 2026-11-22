# Proof certificate format (version 1)

A certificate is one JSON document (optionally gzip-compressed, `.json.gz`) written by `export_certificate`
(`agent/julia/VeryDiffAgentKernel/src/Certificate.jl`) and replayed by the independent checker
`agent/checker/check_certificate.jl`:

```bash
julia --project=agent/julia/VeryDiffAgentKernel agent/checker/check_certificate.jl CERT NET1 NET2 SPEC [--verbose]
# → ACCEPT SAFE|UNSAFE|PARTIAL ...   or   REJECT <reason>
```

A complete small example with explanations: [`examples/example_certificate.jsonc`](examples/example_certificate.jsonc)
(the checkable file is `examples/example_certificate.json`, made by `examples/make_example_certificate.jl`).

The checker trusts nothing in the certificate except what it re-derives: it re-parses the networks and the spec,
recomputes the spec hash, reconstructs every region from the spec box and the cuts, re-runs every propagation
(VeryDiff's transformers and dual bounds, plus its own re-implementations of seeding, contraction and LP-row
construction), re-evaluates every LP bound from the given multipliers, and derives closure itself.
Floats that may be infinite are encoded as the strings `"Infinity"` / `"-Infinity"`.

## Top level

| Key | Meaning |
|---|---|
| `version` | 1 |
| `kernel_version` | kernel that produced it (informational) |
| `spec_hash` | sha256 over length-prefixed (net1 bytes, net2 bytes, spec bytes, canonical property JSON, git tree hash of VeryDiff's `src/`, with `-dirty` if modified). The checker recomputes it from the files it is given and its own VeryDiff checkout. |
| `verydiff_commit`, `verydiff_src_tree` | provenance; `verydiff_src_tree` must equal the checker's |
| `property` | `{"kind": "epsilon", "epsilon": ε}` (only kind supported by the checker) |
| `files` | basenames of net1, net2, spec (informational) |
| `softmax_removed` | whether a final Softmax was dropped (ε refers to the pre-Softmax outputs); must match |
| `options` | `{"csb": bool, "fallback": bool, "lp_relax": bool, "bound_rows": K, "bound_rows_rtol": r}`: consistency tightening z1 = z2 + d on every ReLU layer; VeryDiff's `ONE_SIDED_DIFF_FALLBACK` flag; whether LP leaves may use relaxation rows; VeryDiff's `BOUND_ROWS` / `BOUND_ROWS_RTOL` (bound rows per ReLU layer, 0 = off; missing = 0 / 0.05) |
| `status` | `SAFE`, `UNSAFE` or `PARTIAL` |
| `spec_box` | `{"lo": [...], "hi": [...]}`, must equal the VNNLIB box |
| `active_dims` | 1-based input dimensions with lo < hi |
| `obligations` | `[{"id", "kind": "main"|"lemma", "root", "target": "property"|"lemma:<id>"}]`; exactly one main |
| `nodes` | proof-tree nodes, see below |
| `lemmas` | lemma statements, see below |
| `cex` | `null`, or `{"x", "n1", "n2", "dist"}` for UNSAFE |
| `log` | the action log (`{"t", "action", "args", "rationale"}`), informational |
| `stats` | `nodes`, `auto_nodes`, `kernel_seconds`, `propagations`, `checks`, `inspection_propagations`, `trial_propagations` (informational) |

**UNSAFE** certificates are checked by evaluating both networks at `cex.x` (must lie in the box and violate ε);
the tree is ignored. **SAFE** requires the main root to be closed; **PARTIAL** reports the verified closed
volume fraction: input splits divide a node's volume in proportion to the cut; hidden and relational splits do
not divide it (their children share the region), so such a subtree counts only if it is closed as a whole.

## Nodes

Each node is `[id, parent, side, kind, payload, region]`:

* `parent` 0 for obligation roots. `side`: 0 = root or the single child of an assume node; 1 = lower child;
  2 = upper child.
* `region` = `[lo, hi]` restricted to `active_dims`. Redundant: the checker reconstructs it and requires equality.
* `payload["p"]` (splits and assumes): the node was propagated before being expanded, so its children inherit
  its post-propagation bounds cache; otherwise they inherit its inherited cache unchanged. The checker replays
  that propagation (it costs one propagation) or passes the cache through.

| kind | payload | Meaning |
|---|---|---|
| `"s"` input split | `{"b": "primary", "d": dim, "c": cut, "p"}` | child 1 = [lo, cut], child 2 = [cut, hi] along 1-based input `dim`; lo < cut < hi |
| `"s"` hidden split | `{"b": "relu", "k": layer, "v": "z1"|"z2"|"d", "j": neuron, "c": t, "p"}` | same region; child 1 assumes v_j ≤ t, child 2 assumes v_j ≥ t, at the input of ReLU layer `k` |
| `"s"` relational split | `{"b": "lin", "k": layer, "terms": [[v, j, a], ...], "c": t, "p"}` | same region; child 1 assumes Σ a·v_j ≤ t, child 2 assumes Σ a·v_j ≥ t |
| `"a"` assume | `{"l": lemma, "p"}` | one child (side 0) in which lemma `l` is in scope |
| `"c"` closed | `{"db": bound}` (+ `"lp"` or `"inf"`) | leaf closed by a check, see below |
| `"o"` open | `{}` | leaf without evidence (only in PARTIAL certificates) |

Hidden-split constraints are inherited by every descendant (and by lemma proofs started below them). In every
propagation below the split the checker (1) contracts the shared noise box with the constraint and rescales all
live zonotopes, (2) seeds the bounds cache of layer `k` with the bound, (3) tightens every later ReLU layer's
cached z1/z2/d bounds by single-row weak duality with the constraint rows, and (4) applies z1 = z2 + d
consistency tightening on seeded layers.

With `options.bound_rows` = K > 0, every propagation also collects bound rows: after the tightening of a ReLU layer,
the K cached bounds with the largest gap to the zonotope's own bound (relative to the zonotope interval width, gap >
`bound_rows_rtol`) become rows such as z_j ≤ u_j. They are used, like the constraint rows, for the dual bounds of
all later ReLU layers. The dual bounds and the row selection are VeryDiff's own code
(`src/Transformers/BoundRows.jl`); the checker calls it.

### Closed leaves

* **Plain:** `{"db": d}`. The checker re-propagates the leaf (with its region, inherited cache, lemmas in
  scope and constraints) and runs VeryDiff's ε check (or, in a lemma obligation, checks the lemma's bounds at the
  stop layer); guards: no NaN, no inconsistent cache.
* **Infeasible:** `{"inf": true}`. The checker must find, during its own propagation, a ReLU layer whose bounds
  are empty (lower − upper > 10⁻⁹·(1 + max(|lower|, |upper|)) with finite bounds) or a contraction with no
  solution in the noise box: the node's concrete set is empty.
* **LP evidence:** `{"db": d, "lp": {"lemmas": [ids], "cons": [indices], "relax": [[k, j, type], ...], "rows":
  [{"o": label, "lam": [λ...]}]}}`. The constraint rows are, in this order: the linear claims of the listed
  lemmas (sorted ids, each in scope), the node's split constraints with the listed 1-based indices, and the
  listed relaxation rows (only if `options.lp_relax`). For each objective label (`out:i:+` / `out:i:-` for
  ±(N1 − N2)_i in the main obligation, `spec:k` for lemma claims) whose plain zonotope bound exceeds its threshold,
  `rows` gives λ ≥ 0 (one per constraint row); the checker requires c + ‖g − Aᵀλ‖₁ + λ·r ≤ threshold, where the
  objective is c + g·ε over the shared noise symbols ε ∈ [−1, 1]^m and A ε ≤ r are the rows. This is weak
  duality; no LP solver is involved in checking.

### Relaxation rows (`type`)

All are true relations between the concrete values of unit `j` at ReLU layer `k` (z = input, a = output,
1/2 = N1/N2, d / t = differential input / output), evaluated on the zonotope rows that one witness ε reproduces;
l, u, dl, du are the cached bounds at that layer in the checker's own replay. Chords are taken over
[min(l, 0), max(u, 0)], so they are valid for any sound bounds.

| type | relation |
|---|---|
| `a1>=0`, `a2>=0` | a ≥ 0 |
| `a1>=z1`, `a2>=z2` | a ≥ z |
| `a1<=ch`, `a2<=ch` | a ≤ s (z − l′), s = u′ / (u′ − l′), l′ = min(l, 0), u′ = max(u, 0) |
| `dpre+`, `dpre-` | d = z1 − z2 (one row per direction) |
| `dpost+`, `dpost-` | t = a1 − a2 (one row per direction) |
| `t<=ch` | t ≤ s (d − dl′) (chord of max(0, d)) |
| `t>=ch` | t ≥ dl′ + (1 − s)(d − dl′) (chord of min(0, d)) |

Each row carries an outward allowance of 10⁻¹² · (1 + |rhs| + ‖g‖₁).

## Lemmas

`[{"id", "obligation", "scope_node", "relu_layer", "tensor", "neurons", "dlo", "dhi", ["n1": [lo, hi]],
["n2": [lo, hi]], ["terms": [[v, j, a]...], "rhs"], "status", ["refutation"]}]`

A lemma states bounds (and optionally one linear relation) on the pre-activations at the input of `relu_layer`
for every concrete point of its scope node's region. Its proof is the obligation `obligation`, rooted at a node
with the scope node's region, inherited cache, lemmas and split constraints (lemma chaining). A lemma counts as
proven only if its tree is closed and every lemma it inherits is proven; the checker detects cycles. Inside the
scope, lemma bounds are seeded into the bounds cache and lemma claims may be LP rows.

## Closure

Derived by the checker: a closed leaf is closed if its evidence verifies; a split node is closed if both
children are; an assume node is closed if its child is closed and its lemma is proven. SAFE ⟺ main root closed.
