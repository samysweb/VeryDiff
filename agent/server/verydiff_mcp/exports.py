"""Exports: full data goes to the sandbox's read-only /exports, never into the chat."""
from __future__ import annotations

import json
import time

import numpy as np

from .kernel_client import KernelClient
from .sandbox import Sandbox

WHATS = ("tree", "regions", "bounds", "generators")


def export_to_sandbox(kernel: KernelClient, sandbox: Sandbox, what: str, node_ids: list[int] | None) -> dict:
    if what not in WHATS:
        raise ValueError(f"what must be one of {WHATS}")
    if what != "tree" and not node_ids:
        raise ValueError("node_ids is required for this export")
    stem = f"{what}_{int(time.time() * 1000)}"
    host_json, box_json = sandbox.export_path(stem + ".json")
    r = kernel.call("export_data", {"what": what, "node_ids": node_ids, "path": str(host_json)})
    files = [box_json]
    # Convenience NPZ for numeric exports (converted from the kernel's JSON on the host).
    if what in ("bounds", "generators"):
        data = json.loads(host_json.read_text())
        arrays = {}
        for nid, d in data["nodes"].items():
            if what == "bounds":
                for layer, per in d.items():
                    for k, b in per.items():
                        arrays[f"n{nid}_L{layer}_{k}_lo"] = np.asarray(b["lo"])
                        arrays[f"n{nid}_L{layer}_{k}_hi"] = np.asarray(b["hi"])
            else:
                for k in ("d", "z1", "z2"):
                    arrays[f"n{nid}_{k}_c"] = np.asarray(d[k]["c"])
                    for j, G in enumerate(d[k]["G"]):
                        arrays[f"n{nid}_{k}_G{j}"] = np.asarray(G)
                arrays[f"n{nid}_input_c"] = np.asarray(d["input"]["c"])
                arrays[f"n{nid}_input_G"] = np.asarray(d["input"]["G"])
        host_npz, box_npz = sandbox.export_path(stem + ".npz")
        np.savez_compressed(host_npz, **arrays)
        files.append(box_npz)
    return {"files": files, "bytes": r.get("bytes")}
