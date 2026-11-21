#!/usr/bin/env python3
"""Generate tiny ReLU MLP pairs (ONNX) and VNNLIB box specs for VeryDiff-Agent tests.

Only ops supported by VeryDiff's nn-loader path are used: the same layout as the
ACAS Xu example nets (MatMul(W, x) -> Add(b) -> Relu, 1-D input tensor).
net2 is derived from net1 by weight perturbation and/or magnitude pruning.

Usage: gen_tiny_nets.py OUTDIR
Writes OUTDIR/<case>/{net1.onnx, net2.onnx, spec.vnnlib, meta.json} and OUTDIR/instances.json.
"""
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_mlp(weights, biases, path):
    n_in = weights[0].shape[1]
    n_out = weights[-1].shape[0]
    nodes, inits = [], []
    cur = "X"
    for i, (W, b) in enumerate(zip(weights, biases)):
        inits.append(numpy_helper.from_array(W.astype(np.float64), f"W{i}"))
        inits.append(numpy_helper.from_array(b.astype(np.float64), f"B{i}"))
        nodes.append(helper.make_node("MatMul", [f"W{i}", cur], [f"M{i}"]))
        last = i == len(weights) - 1
        out = "Y" if last else f"H{i}"
        nodes.append(helper.make_node("Add", [f"M{i}", f"B{i}"], [out]))
        if not last:
            nodes.append(helper.make_node("Relu", [out], [f"R{i}"]))
            cur = f"R{i}"
    graph = helper.make_graph(
        nodes, "tiny_mlp",
        [helper.make_tensor_value_info("X", TensorProto.DOUBLE, [n_in])],
        [helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [n_out])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def forward(weights, biases, x):
    h = x
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = W @ h + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def write_spec(lo, hi, n_out, path):
    with open(path, "w") as f:
        for i in range(len(lo)):
            f.write(f"(declare-const X_{i} Real)\n")
        for j in range(n_out):
            f.write(f"(declare-const Y_{j} Real)\n")
        for i in range(len(lo)):
            f.write(f"(assert (>= X_{i} {float(lo[i])!r}))\n")
            f.write(f"(assert (<= X_{i} {float(hi[i])!r}))\n")
        # VeryDiff ignores the output constraint for equivalence checking; a dummy one keeps parsers happy.
        f.write("(assert (>= Y_0 -1000000.0))\n")


def gen_pair(rng, n_in, hidden, n_out, perturb, prune):
    dims = [n_in] + hidden + [n_out]
    W1, B1 = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        W1.append(rng.normal(0, 1.0 / np.sqrt(a), size=(b, a)))
        B1.append(rng.normal(0, 0.1, size=b))
    W2 = [w + perturb * rng.normal(0, 1.0 / np.sqrt(w.shape[1]), size=w.shape) for w in W1]
    B2 = [b + perturb * rng.normal(0, 0.1, size=b.shape) for b in B1]
    if prune > 0:
        for w in W2:
            thr = np.quantile(np.abs(w), prune)
            w[np.abs(w) < thr] = 0.0
    return W1, B1, W2, B2


def sample_maxdiff(W1, B1, W2, B2, lo, hi, rng, n=20000):
    X = rng.uniform(lo, hi, size=(n, len(lo)))
    best = 0.0
    for x in X:
        best = max(best, float(np.max(np.abs(forward(W1, B1, x) - forward(W2, B2, x)))))
    return best


CASES = [
    # name, seed, n_in, hidden, n_out, perturb, prune, box half-width, eps factor (x sampled max), expected
    ("safe_split", 1, 2, [8, 8], 2, 0.02, 0.0, 1.0, 1.5, "SAFE"),
    ("unsafe", 2, 3, [16, 16], 2, 0.05, 0.0, 1.0, 0.5, "UNSAFE"),
    ("safe_hard", 3, 4, [24, 24, 16], 3, 0.03, 0.2, 1.0, 1.15, "SAFE?"),
    ("safe_easy", 4, 2, [8, 8], 2, 0.01, 0.0, 0.1, 3.0, "SAFE"),
    ("safe_mid", 5, 3, [16, 16], 2, 0.02, 0.1, 0.5, 1.3, "SAFE?"),
]


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    instances = []
    for name, seed, n_in, hidden, n_out, perturb, prune, hw, epsf, expected in CASES:
        rng = np.random.default_rng(seed)
        W1, B1, W2, B2 = gen_pair(rng, n_in, hidden, n_out, perturb, prune)
        lo = -hw * np.ones(n_in)
        hi = hw * np.ones(n_in)
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        make_mlp(W1, B1, os.path.join(d, "net1.onnx"))
        make_mlp(W2, B2, os.path.join(d, "net2.onnx"))
        write_spec(lo, hi, n_out, os.path.join(d, "spec.vnnlib"))
        md = sample_maxdiff(W1, B1, W2, B2, lo, hi, rng)
        eps = float(f"{md * epsf:.6g}")
        meta = dict(name=name, n_in=n_in, hidden=hidden, n_out=n_out, perturb=perturb, prune=prune,
                    sampled_max_diff=md, epsilon=eps, expected=expected)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        # paths are relative to the directory holding instances.json
        instances.append(dict(name=name, net1=f"{name}/net1.onnx", net2=f"{name}/net2.onnx",
                              spec=f"{name}/spec.vnnlib", epsilon=eps, expected=expected))
        print(f"{name}: sampled max |N1-N2| = {md:.4g}, eps = {eps}")
    with open(os.path.join(outdir, "instances.json"), "w") as f:
        json.dump(instances, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "agent/tests/nets")
