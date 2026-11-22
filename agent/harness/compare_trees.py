#!/usr/bin/env python3
"""Compare the proof trees of two (or more) certificates: size, depth, split dimensions and cut
positions, manual vs automatic decisions, lemmas.

  python3 agent/harness/compare_trees.py LABEL=CERT [LABEL=CERT ...]
"""
import gzip
import json
import sys
from collections import Counter


def load(p):
    raw = open(p, "rb").read()
    return json.loads(gzip.decompress(raw) if p.endswith(".gz") else raw)


def analyse(C):
    nodes = {e[0]: e for e in C["nodes"]}
    roots = {o["root"]: o for o in C["obligations"]}
    depth = {}
    obl = {}
    for nid in sorted(nodes):
        p = nodes[nid][1]
        depth[nid] = 0 if p == 0 else depth[p] + 1
        obl[nid] = roots[nid]["kind"] if p == 0 else obl[p]
    manual_ids = set()
    for e in C["log"]:
        if e["action"] in ("split",):
            manual_ids.add(e["args"]["node"])
    main = [n for n in nodes.values() if obl[n[0]] == "main"]
    splits = [n for n in main if n[3] == "s"]
    leaves = [n for n in main if n[3] in ("c", "o")]
    act = C["active_dims"]
    lo0 = C["spec_box"]["lo"]; hi0 = C["spec_box"]["hi"]
    def rel_cut(n):
        d = n[4]["d"]; k = act.index(d)
        lo, hi = n[5][0][k], n[5][1][k]
        return round((n[4]["c"] - lo) / (hi - lo), 3)
    top = sorted(splits, key=lambda n: depth[n[0]])[:7]
    return {
        "status": C["status"],
        "nodes_total": len(nodes),
        "main_nodes": len(main),
        "main_leaves": len(leaves),
        "lemma_nodes": len(nodes) - len(main),
        "max_depth": max(depth[n[0]] for n in main),
        "mean_leaf_depth": round(sum(depth[n[0]] for n in leaves) / len(leaves), 2),
        "split_dims": dict(sorted(Counter(n[4]["d"] for n in splits).items())),
        "manual_splits": sum(1 for n in splits if n[0] in manual_ids),
        "non_midpoint_cuts": sum(1 for n in splits if abs(rel_cut(n) - 0.5) > 1e-9),
        "first_splits(depth,dim,rel_cut)": [(depth[n[0]], n[4]["d"], rel_cut(n)) for n in top],
        "lemmas": [(l["id"], l["relu_layer"], len(l["neurons"]), l["status"]) for l in C["lemmas"]],
        "actions": dict(Counter(e["action"] for e in C["log"])),
        "rationales": [f'{e["action"]}: {e["rationale"]}'[:160] for e in C["log"] if e["rationale"]][:12],
    }


def main(args):
    res = {}
    for a in args:
        label, path = a.split("=", 1)
        res[label] = analyse(load(path))
    keys = list(next(iter(res.values())).keys())
    for k in keys:
        if k == "rationales":
            continue
        print(f"{k}:")
        for label, r in res.items():
            print(f"   {label:>10}: {r[k]}")
    for label, r in res.items():
        print(f"\nrationales [{label}]:")
        for x in r["rationales"]:
            print("   -", x)


if __name__ == "__main__":
    main(sys.argv[1:])
