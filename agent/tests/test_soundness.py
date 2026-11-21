#!/usr/bin/env python3
"""Soundness fuzzing of kernel certificates (independent of Julia: numpy forward pass from ONNX).

For SAFE certificates: sample points in every closed leaf of the main obligation and check
|N1(x) − N2(x)|∞ ≤ ε. For lemmas: sample points in the lemma region and check that the
∂ pre-activations at the lemma tensor lie within the stated bounds.

Usage:
  python3 agent/tests/test_soundness.py                  # build certificates via the kernel, then fuzz
  python3 agent/tests/test_soundness.py CERT NET1 NET2   # fuzz an existing certificate
Env: VDA_FUZZ_PER_LEAF (default 10000), VDA_FUZZ_MAX_POINTS (default 2e7 per certificate).
"""
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT / "server"))

PER_LEAF = int(os.environ.get("VDA_FUZZ_PER_LEAF", "10000"))
MAX_POINTS = int(float(os.environ.get("VDA_FUZZ_MAX_POINTS", "2e7")))


class Net:
    """Evaluates MatMul/Add/Relu graphs (the tiny-net layout), returning every tensor."""

    def __init__(self, path):
        m = onnx.load(path)
        self.init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in m.graph.initializer}
        self.nodes = list(m.graph.node)
        self.inp = m.graph.input[0].name
        self.out = m.graph.output[0].name

    def run(self, X):  # X: (n, d)
        vals = {self.inp: X}
        for nd in self.nodes:
            a = [vals.get(i, self.init.get(i)) for i in nd.input]
            if nd.op_type == "MatMul":
                W, x = (a[0], a[1]) if nd.input[0] in self.init else (a[1].T, a[0].T)
                vals[nd.output[0]] = x @ W.T
            elif nd.op_type == "Gemm":
                vals[nd.output[0]] = a[0] @ a[1].T + a[2]
            elif nd.op_type == "Add":
                vals[nd.output[0]] = a[0] + a[1]
            elif nd.op_type == "Relu":
                vals[nd.output[0]] = np.maximum(a[0], 0.0)
            else:
                raise NotImplementedError(nd.op_type)
        return vals


def load(path):
    raw = Path(path).read_bytes()
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def dec(v):
    return np.array([{"Infinity": np.inf, "-Infinity": -np.inf}.get(x, x) if isinstance(x, str) else x for x in v], float)


def fuzz(cert_path, net1, net2, rng=np.random.default_rng(0)):
    C = load(cert_path)
    N1, N2 = Net(net1), Net(net2)
    lo0 = np.array(C["spec_box"]["lo"]); hi0 = np.array(C["spec_box"]["hi"])
    act = np.array(C["active_dims"]) - 1
    eps = C["property"]["epsilon"]
    nodes = {e[0]: e for e in C["nodes"]}

    def box(e):
        lo, hi = lo0.copy(), hi0.copy()
        lo[act] = e[5][0]; hi[act] = e[5][1]
        return lo, hi

    obl_of = {}
    roots = {o["root"]: o for o in C["obligations"]}

    def obligation(nid):
        while nodes[nid][1] != 0:
            nid = nodes[nid][1]
        return roots[nid]

    main_leaves = [e for e in C["nodes"] if e[3] == "c" and obligation(e[0])["kind"] == "main"]
    per = max(10, min(PER_LEAF, MAX_POINTS // max(1, len(main_leaves))))
    worst = 0.0
    viol = 0
    total = 0
    for e in main_leaves:
        lo, hi = box(e)
        X = rng.uniform(lo, hi, size=(per, len(lo)))
        X = np.vstack([X, lo, hi])                     # include two corners
        d = np.max(np.abs(N1.run(X)[N1.out] - N2.run(X)[N2.out]), axis=1)
        worst = max(worst, float(d.max()))
        viol += int((d > eps).sum())
        total += len(X)
    out = {"cert": os.path.basename(cert_path), "status": C["status"], "leaves": len(main_leaves),
           "points": total, "worst_dist": worst, "eps": eps, "violations": viol}
    # lemmas: sample their scope region
    lem_viol = 0
    for L in C["lemmas"]:
        if L["status"] != "proven":
            continue
        lo, hi = box(nodes[L["scope_node"]])
        X = rng.uniform(lo, hi, size=(min(PER_LEAF * 10, 200000), len(lo)))
        h1 = N1.run(X)[L["tensor"]]; h2 = N2.run(X)[L["tensor"]]
        nidx = np.array(L["neurons"]) - 1
        dd = h1[:, nidx] - h2[:, nidx]
        lem_viol += int(((dd < dec(L["dlo"]) - 1e-12) | (dd > dec(L["dhi"]) + 1e-12)).sum())
    out["lemma_violations"] = lem_viol
    out["ok"] = (C["status"] != "SAFE") or (viol == 0 and lem_viol == 0)
    return out


def build_certificates(work):
    from verydiff_mcp.kernel_client import KernelClient
    insts = {i["name"]: i for i in json.loads((AGENT / "tests/nets/instances.json").read_text())}
    k = KernelClient(log_path=Path(work) / "kernel.log", warmup=False)
    k.start()
    certs = []
    try:
        for name in ("safe_easy", "safe_split", "safe_mid", "safe_hard"):
            i = insts[name]
            p = {x: str(AGENT / "tests/nets" / i[x]) for x in ("net1", "net2", "spec")}
            r = k.call("open_session", {**p, "property": {"kind": "epsilon", "epsilon": i["epsilon"]}})
            root = r["root"]
            if name == "safe_mid":
                # use a (true) lemma at the last ReLU layer, obtained from the root's own bounds
                k.call("check", {"node": root})
                lay = r["relu_layers"][-1]
                ib = k.call("inspect_layer", {"node": root, "layer": lay, "neurons": [1, 2, 3, 4]})["input_bounds"]
                pad = 1e-6
                lr = k.call("introduce_lemma", {"node": root, "relu_layer": lay, "neurons": [1, 2, 3, 4],
                                                "dlo": [v - pad for v in ib["lo"]], "dhi": [v + pad for v in ib["hi"]]})
                k.call("prove_lemma", {"lemma": lr["lemma"], "max_nodes": 1000})
                root = lr["child"]
            k.call("auto_expand", {"node": root, "max_nodes": 100000, "max_seconds": 600})
            c = str(Path(work) / f"{name}.json.gz")
            k.call("export_certificate", {"path": c})
            certs.append((c, p["net1"], p["net2"]))
    finally:
        k.close()
    return certs


def main():
    if len(sys.argv) == 4:
        r = fuzz(*sys.argv[1:4])
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    work = tempfile.mkdtemp(prefix="vda_fuzz_")
    ok = True
    for c, n1, n2 in build_certificates(work):
        r = fuzz(c, n1, n2)
        print(json.dumps(r))
        ok &= r["ok"] and r["status"] == "SAFE"
    print("SOUNDNESS FUZZ", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
