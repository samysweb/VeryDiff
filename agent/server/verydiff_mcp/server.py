"""VeryDiff-Agent MCP server.

A thin layer: validates arguments, forwards to the trusted Julia kernel, and keeps
responses short. All soundness logic lives in the kernel.

Run:  python -m verydiff_mcp.server        (with agent/server on PYTHONPATH)
Env:  VDA_WORKDIR        session directory (certificates, sandbox mounts); default ./vda_session
      VDA_SANDBOX        "0" disables the sandbox tools
      VDA_KERNEL_PROJECT path of agent/julia/VeryDiffAgentKernel (default: next to this package)
      VDA_JULIA, VDA_SYSIMAGE, VDA_CALL_TIMEOUT, VDA_WARMUP, VDA_SANDBOX_IMAGE
      VDA_FINAL_CERT     if set, the certificate is exported there when the server exits
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .kernel_client import KernelClient, KernelError, KernelTimeout
from .sandbox import Sandbox, SandboxError
from . import exports as _exports

MAX_RESPONSE = int(os.environ.get("VDA_MAX_RESPONSE", "6000"))
WORKDIR = Path(os.environ.get("VDA_WORKDIR", "vda_session")).resolve()

SOUNDNESS = ("Soundness model: proof actions go to a trusted kernel; SAFE is derived only when every "
             "leaf of the main obligation is closed by a VeryDiff check and every used lemma is proven. "
             "Nothing you compute elsewhere (sandbox, reasoning) counts as evidence.")

mcp = MCPServer(
    name="verydiff-agent",
    instructions=(
        "Drive VeryDiff's differential proof search for ε-equivalence of two ReLU networks. "
        + SOUNDNESS +
        " Typical loop: open_session → check(root) → list_open_leaves → auto_expand with a budget / "
        "split / introduce_lemma+prove_lemma → session_status → export_certificate. "
        "Give a short `rationale` for every proof action. Use the sandbox (no network) for sampling, "
        "local VeryDiff runs and LP experiments; it only informs your next action."),
)

_state: dict[str, Any] = {"kernel": None, "sandbox": None, "session": None, "calls": 0, "t0": time.time()}


_kernel_lock = __import__("threading").Lock()


def kernel() -> KernelClient:
    with _kernel_lock:
        if _state["kernel"] is None:
            WORKDIR.mkdir(parents=True, exist_ok=True)
            _state["kernel"] = KernelClient(log_path=WORKDIR / "kernel.log")
    _state["kernel"].start()
    return _state["kernel"]


def sandbox() -> Sandbox:
    if os.environ.get("VDA_SANDBOX", "1") == "0":
        raise SandboxError("sandbox disabled (VDA_SANDBOX=0)")
    if _state["sandbox"] is None:
        _state["sandbox"] = Sandbox(WORKDIR / "sandbox")
    return _state["sandbox"]


def _shorten(obj: Any) -> Any:
    s = json.dumps(obj, separators=(",", ":"), default=str)
    if len(s) <= MAX_RESPONSE:
        return obj
    return {"truncated": True, "chars": len(s), "head": s[:MAX_RESPONSE],
            "hint": "use limit/offset/neurons, or export_to_sandbox for full data"}


def _call(method: str, **params) -> Any:
    _state["calls"] += 1
    params = {k: v for k, v in params.items() if v is not None}
    try:
        return _shorten(kernel().call(method, params))
    except KernelError as e:
        return {"error": e.type, "message": e.message}
    except KernelTimeout as e:
        return {"error": "timeout", "message": str(e)}


def _rat(r: str | None) -> str:
    if r is not None and not isinstance(r, str):
        raise ValueError("rationale must be a string")
    return (r or "")[:2000]


# ====================================================================== session
@mcp.tool()
def open_session(net1: str, net2: str, spec: str, epsilon: float, csb: bool = False, fallback: bool = False,
                 lp_relax: bool = False) -> Any:
    """Open a proof session for ε-equivalence (max_i |N1(x)_i − N2(x)_i| ≤ ε for all x in the VNNLIB input box).
    The spec is fixed for the session. Returns the root node id and layer info (ReLU layers and widths).
    csb=True tightens every ReLU-input bound with z1 = z2 + d (always done on layers with splits/lemmas).
    fallback=True relaxes one-sided differential ReLU cases with the d-based relaxation when narrower (sound;
    recommended with split_relu, where fixing one phase can otherwise loosen the difference).
    lp_relax=True adds LP-relaxation rows (per-network triangles, identities d = z1 − z2 / t = a1 − a2, differential
    chords) to the final check of every node: much tighter bounds, ~15x the cost of a plain propagation per check."""
    for p in (net1, net2, spec):
        if not Path(p).is_file():
            return {"error": "io", "message": f"file not found: {p}"}
    if not (epsilon >= 0 and epsilon == epsilon and epsilon != float("inf")):
        return {"error": "precondition", "message": "epsilon must be finite and >= 0"}
    paths = {k: str(Path(v).resolve()) for k, v in (("net1", net1), ("net2", net2), ("spec", spec))}
    r = _call("open_session", **paths, property={"kind": "epsilon", "epsilon": float(epsilon)}, csb=bool(csb),
              fallback=bool(fallback), lp_relax=bool(lp_relax))
    if "error" not in r:
        _state["session"] = {**paths, "epsilon": epsilon, "spec_hash": r.get("spec_hash")}
        if os.environ.get("VDA_SANDBOX", "1") != "0":
            try:
                sb = sandbox()
                r["sandbox_files"] = sb.stage_inputs({"net1.onnx": paths["net1"], "net2.onnx": paths["net2"],
                                                      "spec.vnnlib": paths["spec"]})
            except Exception as e:  # sandbox is optional for proving
                r["sandbox_warning"] = str(e)[:300]
    return r


# ====================================================================== proof actions
@mcp.tool()
def check(node: int, rationale: str = "") -> Any:
    """Run VeryDiff on an open leaf (with its bounds cache and in-scope lemmas). If the property holds the
    leaf becomes closed; a reported counterexample is re-evaluated concretely before UNSAFE is declared."""
    return _call("check", node=int(node), rationale=_rat(rationale))


@mcp.tool()
def split(node: int, dim: int, cut: float | None = None, cuts: list[float] | None = None, dim_block: str = "primary",
          rationale: str = "") -> Any:
    """Split an open leaf along input dimension `dim` (1-based) at `cut` (default midpoint; must satisfy lo<cut<hi),
    or k-way at increasing `cuts`. Children exactly partition the parent. dim_block: primary (input box);
    sec1/sec2 (differential perturbation dims) are reserved for robustness properties and rejected for ε."""
    if dim_block not in ("primary", "sec1", "sec2"):
        return {"error": "precondition", "message": "dim_block must be primary|sec1|sec2"}
    return _call("split", node=int(node), dim=int(dim), cut=cut, cuts=cuts, dim_block=dim_block, rationale=_rat(rationale))


@mcp.tool()
def split_relu(node: int, layer: int, neuron: int, var: str = "z1", t: float = 0.0, rationale: str = "") -> Any:
    """Case split an open leaf on a hidden pre-activation at the input of ReLU layer `layer` (1-based neuron):
    var z1 (N1), z2 (N2) or d (N1−N2). Child 1 assumes v <= t, child 2 assumes v >= t; both keep the region.
    Any threshold t is allowed: e.g. bands of the one activation that drives the difference. Every propagation
    below uses the constraint: noise-box contraction, dual-tightened bounds at all later ReLU layers and as LP
    rows in the final check (all re-verified by the checker).
    With t=0 on z1/z2 this fixes the unit's phase (inactive/active) in every propagation below, removing its
    relaxation; the other network's bound is tightened through d. A branch whose bounds become empty closes as
    'infeasible' when checked. Chain two calls (z1 then z2) for a 4-way phase split; unchecked split nodes cost
    no propagation."""
    if var not in ("z1", "z2", "d"):
        return {"error": "precondition", "message": "var must be z1|z2|d"}
    return _call("split_relu", node=int(node), layer=int(layer), neuron=int(neuron), var=var, t=float(t),
                 rationale=_rat(rationale))


@mcp.tool()
def split_lin(node: int, layer: int, terms: list[list], t: float, rationale: str = "") -> Any:
    """Relational case split: child 1 assumes Σ a·v ≤ t, child 2 assumes Σ a·v ≥ t, with terms [[var, neuron, a], ...]
    over pre-activations var ∈ {z1, z2, d} (1-based neurons) at the input of ReLU layer `layer`. On the first ReLU
    layer (affine in the input) this is an arbitrary oblique cut of the input box. Regions are unchanged."""
    return _call("split_lin", node=int(node), layer=int(layer), terms=[[str(v), int(j), float(a)] for v, j, a in terms],
                 t=float(t), rationale=_rat(rationale))


@mcp.tool()
def trial_actions(node: int, actions: list[dict], max_nodes: int = 2000, baseline: bool = True,
                  max_seconds: float = 600.0) -> Any:
    """Dry run: for each candidate expansion of the open leaf `node`, apply it, finish every resulting leaf with
    VeryDiff's best-first search (≤ max_nodes checks) and report the checks needed; then undo it. Actions:
    {"kind": "split", "dim": d, "cut": c?} | {"kind": "split_relu", "layer": k, "neuron": j, "var": "z1|z2|d", "t": t}
    | {"kind": "phase4", "layer": k, "neuron": j}. baseline=True adds plain auto_expand. Proof state unchanged;
    cost is reported as trial propagations. Requires a session without lemmas."""
    return _call("trial_actions", node=int(node), actions=actions, max_nodes=int(max(1, min(max_nodes, 200000))),
                 baseline=bool(baseline), max_seconds=float(max(1.0, min(max_seconds, 3600.0))))


@mcp.tool()
def network_diff_structure(tol: float = 1e-3) -> Any:
    """How N2 differs from N1, per dense layer (1-based units): units/inputs whose weights differ by > tol, units
    knocked out (zero weight row) in either network and their bias, inputs ignored, and the max weight/bias
    difference elsewhere (noise level). Start here to read the 'program diff' (e.g. pruning, fine-tuning)."""
    return _call("network_diff_structure", tol=float(tol))


@mcp.tool()
def difference_attribution(node: int, layer: int, row: int, n_samples: int = 20000, top_k: int = 15,
                           sign: int = 1) -> Any:
    """Sampled decomposition of the N1−N2 difference of row `row` of dense layer `layer` (its pre-activation)
    into per-input-unit terms W1[row,j]·a1_j − W2[row,j]·a2_j, over the node's region and case-split constraints.
    Returns the sampled max difference, the sum of per-term maxima (what abstract interpretation effectively adds
    up), and the units with the largest 'loss' (own max − value at the arg-max point) with phase frequencies:
    candidates for case distinctions. sign=-1 analyses the lower side. Inspection only, not a bound."""
    return _call("difference_attribution", node=int(node), layer=int(layer), row=int(row),
                 n_samples=int(max(100, min(n_samples, 200000))), top_k=int(top_k), sign=1 if sign >= 0 else -1)


@mcp.tool()
def relu_scores(node: int, top_k: int = 10, rows: list[int] | None = None) -> Any:
    """Rank unstable ReLU units at `node` by the mass of their relaxation generator in the output ∂Z on the
    critical output rows (those whose bound exceeds ε, or `rows`). Each entry gives layer, var (z1/z2/d), neuron,
    score, its share of the row's generator mass and the cached z1/z2/d bounds. Candidates for split_relu.
    Direct mass only: N1 (z1) relaxations reach ∂Z indirectly and typically score 0. One propagation."""
    return _call("relu_scores", node=int(node), top_k=int(max(1, min(top_k, 500))), rows=rows)


@mcp.tool()
def auto_expand(node: int, max_nodes: int = 200, max_seconds: float = 30.0, heuristic: str = "verydiff",
                per_leaf_max_nodes: int | None = None, rationale: str = "") -> Any:
    """Run VeryDiff's branch-and-bound on the open leaves under `node` (heuristic: verydiff|widest).
    Default: one best-first search over all of them with `max_nodes` checks in total.
    With `per_leaf_max_nodes`: an independent search on EACH open leaf under `node` (e.g. "auto on all branches
    after my manual splits"), `max_nodes` then caps the total. Every split/closure is recorded in the tree."""
    return _call("auto_expand", node=int(node), max_nodes=int(max(1, min(max_nodes, 200000))),
                 per_leaf_max_nodes=None if per_leaf_max_nodes is None else int(max(1, per_leaf_max_nodes)),
                 max_seconds=float(max(0.1, min(max_seconds, 3600))), heuristic=heuristic, rationale=_rat(rationale))


@mcp.tool()
def batch(actions: list[dict], stop_on_error: bool = True) -> Any:
    """Run several proof/inspection actions in one call, in order, e.g. a split plan followed by auto_expand.
    Each action is {"method": <tool name>, "params": {...}} with the same parameters as the individual tool and is
    checked by the kernel exactly as if sent alone. A node parameter may reference children of an earlier split in
    the batch: "$k.children[i]" (k = 1-based action index). Allowed: check, split, auto_expand, introduce_lemma,
    prove_lemma, retract_lemma, refute_lemma, prune, submit_counterexample, and read-only inspection tools."""
    if not isinstance(actions, list) or not actions or len(actions) > 500:
        return {"error": "precondition", "message": "actions must be a non-empty list (max 500)"}
    for a in actions:
        if not isinstance(a, dict) or "method" not in a:
            return {"error": "precondition", "message": "each action needs a 'method' (and optional 'params')"}
        if "rationale" in a.get("params", {}):
            a["params"]["rationale"] = _rat(a["params"]["rationale"])
    return _call("batch", actions=actions, stop_on_error=bool(stop_on_error))


@mcp.tool()
def introduce_lemma(node: int, relu_layer: int, neurons: list[int], dlo: list[float | str], dhi: list[float | str],
                    n1_bounds: list[list[float | str]] | None = None, n2_bounds: list[list[float | str]] | None = None,
                    rationale: str = "") -> Any:
    """Assume bounds on ∂ = N1 − N2 pre-activations at the input of ReLU layer `relu_layer` (neurons 1-based) over
    region(node). The node becomes assume(L) with one child where the bounds seed VeryDiff's relaxation.
    The lemma becomes its own proof obligation; SAFE requires it to be proven (prove_lemma). Its proof may use the
    lemmas already in scope at `node` (lemma chains, e.g. layer 3 → layer 6 → layer 9). Use "inf"/"-inf" for
    unbounded entries. Optional n1_bounds/n2_bounds = [lo_list, hi_list] for the individual networks."""
    return _call("introduce_lemma", node=int(node), relu_layer=int(relu_layer), neurons=[int(n) for n in neurons],
                 dlo=dlo, dhi=dhi, n1_bounds=n1_bounds, n2_bounds=n2_bounds, rationale=_rat(rationale))


@mcp.tool()
def introduce_phase_lemma(node: int, relu_layer: int, neurons: list[int], net1: str = "any", net2: str = "any",
                          rationale: str = "") -> Any:
    """Assume that units `neurons` (1-based) of ReLU layer `relu_layer` are inactive ("off": pre-activation ≤ 0)
    or active ("on": ≥ 0) on region(node), in net1 and/or net2 ("any" = no claim). Cheaper to prove than interval
    bounds (one-sided, single network). Same proof/scope rules as introduce_lemma (chainable). Note: VeryDiff picks
    the ReLU case from fresh bounds, so for a unit that is unstable in both networks the ∂ relaxation is only
    tightened through N1/N2, not made exact."""
    for p in (net1, net2):
        if p not in ("off", "on", "any"):
            return {"error": "precondition", "message": "phase must be off|on|any"}
    return _call("introduce_phase_lemma", node=int(node), relu_layer=int(relu_layer), neurons=[int(n) for n in neurons],
                 net1=net1, net2=net2, rationale=_rat(rationale))


@mcp.tool()
def introduce_relational_lemma(node: int, relu_layer: int, terms: list[list], rhs: float | str, rationale: str = "") -> Any:
    """Assume a linear relation Σ coeff·v ≤ rhs over pre-activations of ReLU layer `relu_layer` on region(node).
    terms = [[v, neuron, coeff], ...] with v ∈ "d" (net1−net2), "z1" (net1), "z2" (net2), neurons 1-based.
    It does not change VeryDiff's relaxation; instead every check below the node bounds the outputs by an LP over
    the zonotope's noise symbols with all in-scope lemmas as linear constraints (interval/phase lemmas are used the
    same way). The LP bound is certified by weak duality and replayed by the checker. Prove it with prove_lemma."""
    if not isinstance(terms, list) or not terms:
        return {"error": "precondition", "message": "terms must be a non-empty list of [v, neuron, coeff]"}
    return _call("introduce_relational_lemma", node=int(node), relu_layer=int(relu_layer), terms=terms, rhs=rhs,
                 rationale=_rat(rationale))


@mcp.tool()
def trial_lemma(node: int, relu_layer: int, neurons: list[int] | None = None, kind: str = "phase", net1: str = "any",
                net2: str = "any", terms: list[list] | None = None, rhs: float | str | None = None,
                dlo: list[float | str] | None = None, dhi: list[float | str] | None = None,
                n1_bounds: list[list[float | str]] | None = None, n2_bounds: list[list[float | str]] | None = None,
                lemma_max_nodes: int = 2000, main_max_nodes: int = 2000, max_seconds: float = 120.0,
                baseline: bool = True, rationale: str = "") -> Any:
    """DRY RUN (changes nothing, not counted as proof cost): introduce a phase lemma (kind="phase", net1/net2) or an
    interval lemma (kind="interval", dlo/dhi/n1_bounds/n2_bounds) or a relational lemma (kind="relational",
    terms/rhs) at open leaf `node`, prove it within
    lemma_max_nodes, auto_expand the assume child within main_max_nodes, and (baseline=True) auto_expand `node`
    without the lemma with the same budget. Returns exact nodes/checks for each part and net_saving_nodes. Use it to
    decide WHERE a lemma pays off before introducing it for real."""
    if kind not in ("phase", "interval", "relational"):
        return {"error": "precondition", "message": "kind must be phase|interval|relational"}
    return _call("trial_lemma", node=int(node), relu_layer=int(relu_layer), neurons=[int(n) for n in (neurons or [])],
                 kind=kind, terms=terms, rhs=rhs,
                 net1=net1, net2=net2, dlo=dlo, dhi=dhi, n1_bounds=n1_bounds, n2_bounds=n2_bounds,
                 lemma_max_nodes=int(lemma_max_nodes), main_max_nodes=int(main_max_nodes),
                 max_seconds=float(max_seconds), baseline=bool(baseline), rationale=_rat(rationale))


@mcp.tool()
def prove_lemma(lemma: int, max_nodes: int = 200, max_seconds: float = 30.0, rationale: str = "") -> Any:
    """Run auto_expand on the lemma's proof tree (you can also drive it manually with check/split/auto_expand
    on its nodes; see session_status for its root)."""
    return _call("prove_lemma", lemma=int(lemma), max_nodes=int(max_nodes), max_seconds=float(max_seconds),
                 rationale=_rat(rationale))


@mcp.tool()
def retract_lemma(lemma: int, rationale: str = "") -> Any:
    """Delete a lemma: its assume node becomes an open leaf again and everything that could depend on it is discarded."""
    return _call("retract_lemma", lemma=int(lemma), rationale=_rat(rationale))


@mcp.tool()
def refute_lemma(lemma: int, x: list[float], rationale: str = "") -> Any:
    """Mark a lemma refuted if the kernel confirms that input `x` (inside the lemma's region) violates it."""
    return _call("refute_lemma", lemma=int(lemma), x=[float(v) for v in x], rationale=_rat(rationale))


@mcp.tool()
def prune(node: int, rationale: str = "") -> Any:
    """Backtrack: collapse the subtree under `node` back to an open leaf (discards evidence only)."""
    return _call("prune", node=int(node), rationale=_rat(rationale))


@mcp.tool()
def submit_counterexample(x: list[float], rationale: str = "") -> Any:
    """Propose a counterexample input. The kernel checks it lies in the spec box and violates the property concretely."""
    return _call("submit_counterexample", x=[float(v) for v in x], rationale=_rat(rationale))


@mcp.tool()
def export_certificate(path: str | None = None) -> Any:
    """Write the proof certificate (status SAFE/UNSAFE/PARTIAL; replayable by the independent checker).
    Default path: <workdir>/certificate.json(.gz if path ends in .gz)."""
    p = Path(path) if path else WORKDIR / "certificate.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return _call("export_certificate", path=str(p.resolve()))


# ====================================================================== inspection (read-only)
@mcp.tool()
def session_status() -> Any:
    """Verdict, open/closed leaf counts, closed volume fraction, lemmas and their status, kernel time."""
    return _call("session_status")


@mcp.tool()
def list_layers() -> Any:
    """Layers of the differential network: index, type, width, whether ReLU (lemma targets are ReLU layers)."""
    return _call("list_layers")


@mcp.tool()
def list_open_leaves(sort_by: str = "distance_bound", limit: int = 10, offset: int = 0,
                     obligation: int | None = None, under: int | None = None) -> Any:
    """Open leaves of the main obligation (or of `obligation`, or under node `under`), sorted by
    distance_bound | volume | depth."""
    return _call("list_open_leaves", sort_by=sort_by, limit=int(max(1, min(limit, 100))), offset=int(offset),
                 obligation=obligation, under=under)


@mcp.tool()
def get_node(node: int) -> Any:
    """Region (non-degenerate dims), path, lemmas in scope, last check result, kind/children of a node."""
    return _call("get_node", node=int(node))


@mcp.tool()
def inspect_layer(node: int, layer: int, which: str = "d", neurons: list[int] | None = None,
                  detail: str = "bounds", topk: int = 5) -> Any:
    """Re-propagate `node` and return bounds at `layer` (which: d=N1−N2, z1, z2). For ReLU layers also input
    bounds, phase-selector counts and unstable-neuron counts. detail=generators_topk needs ≤8 neurons."""
    return _call("inspect_layer", node=int(node), layer=int(layer), which=which, neurons=neurons, detail=detail,
                 topk=int(topk))


@mcp.tool()
def output_bounds(node: int, neurons: list[int] | None = None) -> Any:
    """Output bounds of ∂, N1, N2 for `node` and the resulting distance bound."""
    return _call("output_bounds", node=int(node), neurons=neurons)


@mcp.tool()
def split_scores(node: int, top_k: int = 5) -> Any:
    """VeryDiff's own split-heuristic scores per input dimension for `node` (an instrument, not a rule)."""
    return _call("split_scores", node=int(node), top_k=int(top_k))


@mcp.tool()
def cache_bounds(node: int, relu_layer: int, neurons: list[int] | None = None) -> Any:
    """Current bounds cache at a ReLU layer for `node`: the bounds the relaxation actually uses."""
    return _call("cache_bounds", node=int(node), relu_layer=int(relu_layer), neurons=neurons)


@mcp.tool()
def read_weights(layer: int, which: str = "diff", rows: list[int] | None = None, cols: list[int] | None = None,
                 max_entries: int = 400, digits: int = 4) -> Any:
    """Read an affine layer's weights as text: which = net1 | net2 | diff (net1 − net2). Always includes a summary
    of where the difference lives (nonzero rows/cols, norms, top singular values). Select rows/cols for large layers.
    For a Linear layer followed by an AddConst layer (ONNX MatMul+Add), `b` includes that AddConst bias (see b_note)."""
    return _call("read_weights", layer=int(layer), which=which, rows=rows, cols=cols,
                 max_entries=int(max(1, min(max_entries, 2000))), digits=int(max(1, min(digits, 17))))


@mcp.tool()
def eval_point(x: list[float], hidden: bool = False) -> Any:
    """Concretely evaluate both networks at x (outputs, distance; hidden=True adds ReLU pre-activations)."""
    return _call("eval_point", x=[float(v) for v in x], hidden=bool(hidden))


# ====================================================================== sandbox
def _sb(fn, *a, **k):
    try:
        return _shorten(fn(*a, **k))
    except (SandboxError, ValueError) as e:
        return {"error": "sandbox", "message": str(e)[:500]}
    except KernelError as e:
        return {"error": e.type, "message": e.message}


@mcp.tool()
def sandbox_exec(cmd: str, timeout_s: int = 120) -> Any:
    """Run a shell command in the persistent sandbox container (no network; Julia+VeryDiff CLI at
    /opt/verydiff, python3 with numpy/onnx/onnxruntime). /nets and /exports are read-only, /scratch is writable.
    Output is truncated. Results here are NOT evidence."""
    return _sb(lambda: sandbox().exec(cmd, timeout_s))


@mcp.tool()
def sandbox_write_file(path: str, content: str) -> Any:
    """Write a text file under /scratch in the sandbox."""
    return _sb(lambda: sandbox().write_file(path, content))


@mcp.tool()
def export_to_sandbox(what: str, node_ids: list[int] | None = None) -> Any:
    """Write full data to the sandbox's /exports as JSON (+NPZ): what = tree | regions | bounds | generators."""
    return _sb(lambda: _exports.export_to_sandbox(kernel(), sandbox(), what, node_ids))


@mcp.tool()
def sandbox_reset() -> Any:
    """Recreate the sandbox container (wipes /scratch)."""
    return _sb(lambda: sandbox().reset())


def main():
    if os.environ.get("VDA_EAGER_KERNEL", "1") != "0":
        # start Julia (+ warm-up) in the background so the first tool call does not hit client timeouts
        import threading
        threading.Thread(target=kernel, daemon=True).start()
    try:
        mcp.run("stdio")
    finally:
        final = os.environ.get("VDA_FINAL_CERT")
        if final and _state["kernel"] and _state["session"]:
            try:  # harness fallback: always leave a certificate behind
                _state["kernel"].call("export_certificate", {"path": str(Path(final).resolve())}, timeout=300)
            except Exception:
                pass
        if _state["kernel"]:
            _state["kernel"].close()
        if _state["sandbox"]:
            _state["sandbox"].stop()


if __name__ == "__main__":
    main()
