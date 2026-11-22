# MCP tools (`agent/server/verydiff_mcp/server.py`)

The MCP server is a thin client of the Julia kernel (NDJSON RPC). Every proof action is validated by the kernel; no tool can mark a node
closed or a lemma proven. Indices are 1-based (input dimensions, neurons, output rows). Layers are the kernel's layer indices as
returned by `list_layers` (e.g. MNIST 3×100: dense 1, 4, 7, 10; ReLU 3, 6, 9). Variables at a ReLU input: `z1` (N1), `z2` (N2),
`d` (N1 − N2). This file is generated from the tool docstrings; the docstring is the authoritative description.

Server environment: `VDA_WORKDIR`, `VDA_SANDBOX=0|1`, `VDA_SANDBOX_IMAGE`, `VDA_SYSIMAGE`, `VDA_CALL_TIMEOUT`, `VDA_WARMUP`,
`VDA_FINAL_CERT` (certificate written on exit), `VDA_MAX_RESPONSE`, `VDA_MAX_AUTO_NODES` (cap for `auto_expand`/`trial_actions`).

## Session and proof actions

### `open_session(net1: str, net2: str, spec: str, epsilon: float, csb: bool = False, fallback: bool = False, lp_relax: bool = False)`

Open a proof session for ε-equivalence (max_i |N1(x)_i − N2(x)_i| ≤ ε for all x in the VNNLIB input box).
The spec is fixed for the session. Returns the root node id and layer info (ReLU layers and widths).
csb=True tightens every ReLU-input bound with z1 = z2 + d (always done on layers with splits/lemmas).
fallback=True relaxes one-sided differential ReLU cases with the d-based relaxation when narrower (sound;
recommended with split_relu, where fixing one phase can otherwise loosen the difference).
lp_relax=True adds LP-relaxation rows (per-network triangles, identities d = z1 − z2 / t = a1 − a2, differential
chords) to the final check of every node: much tighter bounds, ~15x the cost of a plain propagation per check.

### `check(node: int, rationale: str = '')`

Run VeryDiff on an open leaf (with its bounds cache and in-scope lemmas). If the property holds the
leaf becomes closed; a reported counterexample is re-evaluated concretely before UNSAFE is declared.

### `split(node: int, dim: int, cut: float | None = None, cuts: list[float] | None = None, dim_block: str = 'primary', rationale: str = '')`

Split an open leaf along input dimension `dim` (1-based) at `cut` (default midpoint; must satisfy lo<cut<hi),
or k-way at increasing `cuts`. Children exactly partition the parent. dim_block: primary (input box);
sec1/sec2 (differential perturbation dims) are reserved for robustness properties and rejected for ε.

### `split_relu(node: int, layer: int, neuron: int, var: str = 'z1', t: float = 0.0, rationale: str = '')`

Case split an open leaf on a hidden pre-activation at the input of ReLU layer `layer` (1-based neuron):
var z1 (N1), z2 (N2) or d (N1−N2). Child 1 assumes v <= t, child 2 assumes v >= t; both keep the region.
Any threshold t is allowed: e.g. bands of the one activation that drives the difference. Every propagation
below uses the constraint: dual-tightened bounds at all later ReLU layers, an infeasibility test and LP
rows in the final check (all re-verified by the checker).
With t=0 on z1/z2 this fixes the unit's phase (inactive/active) in every propagation below, removing its
relaxation; the other network's bound is tightened through d. A branch whose bounds become empty closes as
'infeasible' when checked. Chain two calls (z1 then z2) for a 4-way phase split; unchecked split nodes cost
no propagation.

### `split_lin(node: int, layer: int, terms: list[list], t: float, rationale: str = '')`

Relational case split: child 1 assumes Σ a·v ≤ t, child 2 assumes Σ a·v ≥ t, with terms [[var, neuron, a], ...]
over pre-activations var ∈ {z1, z2, d} (1-based neurons) at the input of ReLU layer `layer`. On the first ReLU
layer (affine in the input) this is an arbitrary oblique cut of the input box. Regions are unchanged.

### `auto_expand(node: int, max_nodes: int = 200, max_seconds: float = 30.0, heuristic: str = 'verydiff', per_leaf_max_nodes: int | None = None, rationale: str = '')`

Run VeryDiff's branch-and-bound on the open leaves under `node` (heuristic: verydiff|widest).
Default: one best-first search over all of them with `max_nodes` checks in total.
With `per_leaf_max_nodes`: an independent search on EACH open leaf under `node` (e.g. "auto on all branches
after my manual splits"), `max_nodes` then caps the total. Every split/closure is recorded in the tree.

### `batch(actions: list[dict], stop_on_error: bool = True)`

Run several proof/inspection actions in one call, in order, e.g. a split plan followed by auto_expand.
Each action is {"method": <tool name>, "params": {...}} with the same parameters as the individual tool and is
checked by the kernel exactly as if sent alone. A node parameter may reference children of an earlier split in
the batch: "$k.children[i]" (k = 1-based action index). Allowed: check, split, auto_expand, introduce_lemma,
prove_lemma, retract_lemma, refute_lemma, prune, submit_counterexample, and read-only inspection tools.

### `prune(node: int, rationale: str = '')`

Backtrack: collapse the subtree under `node` back to an open leaf (discards evidence only).

### `submit_counterexample(x: list[float], rationale: str = '')`

Propose a counterexample input. The kernel checks it lies in the spec box and violates the property concretely.

### `export_certificate(path: str | None = None)`

Write the proof certificate (status SAFE/UNSAFE/PARTIAL; replayable by the independent checker).
Default path: <workdir>/certificate.json(.gz if path ends in .gz).

## Lemmas

### `introduce_lemma(node: int, relu_layer: int, neurons: list[int], dlo: list[float | str], dhi: list[float | str], n1_bounds: list[list[float | str]] | None = None, n2_bounds: list[list[float | str]] | None = None, rationale: str = '')`

Assume bounds on ∂ = N1 − N2 pre-activations at the input of ReLU layer `relu_layer` (neurons 1-based) over
region(node). The node becomes assume(L) with one child where the bounds seed VeryDiff's relaxation.
The lemma becomes its own proof obligation; SAFE requires it to be proven (prove_lemma). Its proof may use the
lemmas already in scope at `node` (lemma chains, e.g. layer 3 → layer 6 → layer 9). Use "inf"/"-inf" for
unbounded entries. Optional n1_bounds/n2_bounds = [lo_list, hi_list] for the individual networks.

### `introduce_phase_lemma(node: int, relu_layer: int, neurons: list[int], net1: str = 'any', net2: str = 'any', rationale: str = '')`

Assume that units `neurons` (1-based) of ReLU layer `relu_layer` are inactive ("off": pre-activation ≤ 0)
or active ("on": ≥ 0) on region(node), in net1 and/or net2 ("any" = no claim). Cheaper to prove than interval
bounds (one-sided, single network). Same proof/scope rules as introduce_lemma (chainable). Note: VeryDiff picks
the ReLU case from fresh bounds, so for a unit that is unstable in both networks the ∂ relaxation is only
tightened through N1/N2, not made exact.

### `introduce_relational_lemma(node: int, relu_layer: int, terms: list[list], rhs: float | str, rationale: str = '')`

Assume a linear relation Σ coeff·v ≤ rhs over pre-activations of ReLU layer `relu_layer` on region(node).
terms = [[v, neuron, coeff], ...] with v ∈ "d" (net1−net2), "z1" (net1), "z2" (net2), neurons 1-based.
It does not change VeryDiff's relaxation; instead every check below the node bounds the outputs by an LP over
the zonotope's noise symbols with all in-scope lemmas as linear constraints (interval/phase lemmas are used the
same way). The LP bound is certified by weak duality and replayed by the checker. Prove it with prove_lemma.

### `prove_lemma(lemma: int, max_nodes: int = 200, max_seconds: float = 30.0, rationale: str = '')`

Run auto_expand on the lemma's proof tree (you can also drive it manually with check/split/auto_expand
on its nodes; see session_status for its root).

### `retract_lemma(lemma: int, rationale: str = '')`

Delete a lemma: its assume node becomes an open leaf again and everything that could depend on it is discarded.

### `refute_lemma(lemma: int, x: list[float], rationale: str = '')`

Mark a lemma refuted if the kernel confirms that input `x` (inside the lemma's region) violates it.

## Dry runs (planning; not proof cost)

### `trial_actions(node: int, actions: list[dict], max_nodes: int = 2000, baseline: bool = True, max_seconds: float = 600.0)`

Dry run: for each candidate expansion of the open leaf `node`, apply it, finish every resulting leaf with
VeryDiff's best-first search (≤ max_nodes checks) and report the checks needed; then undo it. Actions:
{"kind": "split", "dim": d, "cut": c?} | {"kind": "split_relu", "layer": k, "neuron": j, "var": "z1|z2|d", "t": t}
| {"kind": "phase4", "layer": k, "neuron": j}. baseline=True adds plain auto_expand. Proof state unchanged;
cost is reported as trial propagations. Requires a session without lemmas.

### `trial_lemma(node: int, relu_layer: int, neurons: list[int] | None = None, kind: str = 'phase', net1: str = 'any', net2: str = 'any', terms: list[list] | None = None, rhs: float | str | None = None, dlo: list[float | str] | None = None, dhi: list[float | str] | None = None, n1_bounds: list[list[float | str]] | None = None, n2_bounds: list[list[float | str]] | None = None, lemma_max_nodes: int = 2000, main_max_nodes: int = 2000, max_seconds: float = 120.0, baseline: bool = True, rationale: str = '')`

DRY RUN (changes nothing, not counted as proof cost): introduce a phase lemma (kind="phase", net1/net2) or an
interval lemma (kind="interval", dlo/dhi/n1_bounds/n2_bounds) or a relational lemma (kind="relational",
terms/rhs) at open leaf `node`, prove it within
lemma_max_nodes, auto_expand the assume child within main_max_nodes, and (baseline=True) auto_expand `node`
without the lemma with the same budget. Returns exact nodes/checks for each part and net_saving_nodes. Use it to
decide WHERE a lemma pays off before introducing it for real.

## Inspection (read-only; propagations counted as inspection)

### `session_status()`

Verdict, open/closed leaf counts, closed volume fraction, lemmas and their status, kernel time.

### `list_layers()`

Layers of the differential network: index, type, width, whether ReLU (lemma targets are ReLU layers).

### `list_open_leaves(sort_by: str = 'distance_bound', limit: int = 10, offset: int = 0, obligation: int | None = None, under: int | None = None)`

Open leaves of the main obligation (or of `obligation`, or under node `under`), sorted by
distance_bound | volume | depth.

### `get_node(node: int)`

Region (non-degenerate dims), path, lemmas in scope, last check result, kind/children of a node.

### `inspect_layer(node: int, layer: int, which: str = 'd', neurons: list[int] | None = None, detail: str = 'bounds', topk: int = 5)`

Re-propagate `node` and return bounds at `layer` (which: d=N1−N2, z1, z2). For ReLU layers also input
bounds, phase-selector counts and unstable-neuron counts. detail=generators_topk needs ≤8 neurons.

### `output_bounds(node: int, neurons: list[int] | None = None)`

Output bounds of ∂, N1, N2 for `node` and the resulting distance bound.

### `cache_bounds(node: int, relu_layer: int, neurons: list[int] | None = None)`

Current bounds cache at a ReLU layer for `node`: the bounds the relaxation actually uses.

### `split_scores(node: int, top_k: int = 5)`

VeryDiff's own split-heuristic scores per input dimension for `node` (an instrument, not a rule).

### `relu_scores(node: int, top_k: int = 10, rows: list[int] | None = None)`

Rank unstable ReLU units at `node` by the mass of their relaxation generator in the output ∂Z on the
critical output rows (those whose bound exceeds ε, or `rows`). Each entry gives layer, var (z1/z2/d), neuron,
score, its share of the row's generator mass and the cached z1/z2/d bounds. Candidates for split_relu.
Direct mass only: N1 (z1) relaxations reach ∂Z indirectly and typically score 0. One propagation.

### `read_weights(layer: int, which: str = 'diff', rows: list[int] | None = None, cols: list[int] | None = None, max_entries: int = 400, digits: int = 4)`

Read an affine layer's weights as text: which = net1 | net2 | diff (net1 − net2). Always includes a summary
of where the difference lives (nonzero rows/cols, norms, top singular values). Select rows/cols for large layers.
For a Linear layer followed by an AddConst layer (ONNX MatMul+Add), `b` includes that AddConst bias (see b_note).

### `network_diff_structure(tol: float = 0.001)`

How N2 differs from N1, per dense layer (1-based units): units/inputs whose weights differ by > tol, units
knocked out (zero weight row) in either network and their bias, inputs ignored, and the max weight/bias
difference elsewhere (noise level). Start here to read the 'program diff' (e.g. pruning, fine-tuning).

### `difference_attribution(node: int, layer: int, row: int, n_samples: int = 20000, top_k: int = 15, sign: int = 1)`

Sampled decomposition of the N1−N2 difference of row `row` of dense layer `layer` (its pre-activation)
into per-input-unit terms W1[row,j]·a1_j − W2[row,j]·a2_j, over the node's region and case-split constraints.
Returns the sampled max difference, the sum of per-term maxima (what abstract interpretation effectively adds
up), and the units with the largest 'loss' (own max − value at the arg-max point) with phase frequencies:
candidates for case distinctions. sign=-1 analyses the lower side. Inspection only, not a bound.

### `eval_point(x: list[float], hidden: bool = False)`

Concretely evaluate both networks at x (outputs, distance; hidden=True adds ReLU pre-activations).

## Sandbox (no network; never evidence)

### `sandbox_exec(cmd: str, timeout_s: int = 120)`

Run a shell command in the persistent sandbox container (no network; Julia+VeryDiff CLI at
/opt/verydiff, python3 with numpy/onnx/onnxruntime). /nets and /exports are read-only, /scratch is writable.
Output is truncated. Results here are NOT evidence.

### `sandbox_write_file(path: str, content: str)`

Write a text file under /scratch in the sandbox.

### `export_to_sandbox(what: str, node_ids: list[int] | None = None)`

Write full data to the sandbox's /exports as JSON (+NPZ): what = tree | regions | bounds | generators.

### `sandbox_reset()`

Recreate the sandbox container (wipes /scratch).
