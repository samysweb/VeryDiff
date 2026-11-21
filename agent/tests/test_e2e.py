#!/usr/bin/env python3
"""End-to-end: MCP client -> verydiff_mcp server -> Julia kernel -> certificate -> independent checker.

Usage: python3 agent/tests/test_e2e.py [--sandbox]
Prints first-call latency and the checker verdicts. Exit code 0 on success.
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

AGENT = Path(__file__).resolve().parents[1]
NETS = AGENT / "tests" / "nets"
KERNEL = AGENT / "julia" / "VeryDiffAgentKernel"
CHECKER = AGENT / "checker" / "check_certificate.jl"


def inst(name):
    for i in json.loads((NETS / "instances.json").read_text()):
        if i["name"] == name:
            return {**i, **{k: str(NETS / i[k]) for k in ("net1", "net2", "spec")}}
    raise KeyError(name)


def payload(res):
    if getattr(res, "structured_content", None):
        sc = res.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    txt = "".join(getattr(c, "text", "") for c in res.content)
    try:
        return json.loads(txt)
    except Exception:
        return txt


def run_checker(cert, i):
    r = subprocess.run(["julia", f"--project={KERNEL}", str(CHECKER), cert, i["net1"], i["net2"], i["spec"]],
                       capture_output=True, text=True, env={**os.environ, "VERYDIFF_NO_GUROBI": "1"})
    return (r.stdout.strip().splitlines() or ["(no output)"])[-1]


async def main(use_sandbox: bool):
    work = Path(tempfile.mkdtemp(prefix="vda_e2e_"))
    env = {**os.environ, "PYTHONPATH": str(AGENT / "server"), "VDA_WORKDIR": str(work),
           "VDA_SANDBOX": "1" if use_sandbox else "0", "VERYDIFF_NO_GUROBI": "1"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "verydiff_mcp.server"], env=env)
    ok = True
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = sorted(t.name for t in tools.tools)
            print("tools:", len(names), names)

            async def call(name, **args):
                t = time.time()
                res = payload(await s.call_tool(name, args, read_timeout_seconds=1800))
                print(f"  {name}({json.dumps(args)[:80]}) -> {time.time() - t:.2f}s {json.dumps(res)[:200]}")
                return res

            # ---- SAFE instance, manual + auto actions
            i = inst("safe_mid")
            t0 = time.time()
            o = await call("open_session", net1=i["net1"], net2=i["net2"], spec=i["spec"], epsilon=i["epsilon"])
            print(f"first call (kernel start + warm-up + open_session): {time.time() - t0:.1f}s")
            root = o["root"]
            await call("check", node=root, rationale="look at the root")
            await call("split_scores", node=root, top_k=3)
            sp = await call("split", node=root, dim=1, rationale="dim 1 has the highest score")
            await call("list_layers")
            await call("inspect_layer", node=sp["children"][0], layer=o["relu_layers"][0])
            await call("auto_expand", node=root, max_nodes=5000, max_seconds=120, rationale="finish with VeryDiff")
            st = await call("session_status")
            ok &= st["verdict"] == "SAFE"
            cert = str(work / "safe_mid.json")
            await call("export_certificate", path=cert)
            v = run_checker(cert, i)
            print("checker:", v)
            ok &= v.startswith("ACCEPT SAFE")

            # ---- errors are structured
            e = await call("split", node=root, dim=1)
            ok &= isinstance(e, dict) and e.get("error") == "precondition"

            if use_sandbox:
                x = await call("sandbox_exec", cmd="curl -sS --max-time 5 https://example.com || echo NO_NET; "
                                                   "touch /exports/x 2>&1 || echo EXPORTS_RO; ls /nets", timeout_s=30)
                ok &= "NO_NET" in x["stdout"] and "EXPORTS_RO" in x["stdout"]
                ex = await call("export_to_sandbox", what="tree")
                y = await call("sandbox_exec", cmd=f"python3 -c \"import json;d=json.load(open('{ex['files'][0]}'));print(len(d['nodes']))\"")
                ok &= y["exit_code"] == 0

            # ---- UNSAFE instance (new session replaces the old one)
            i = inst("unsafe")
            o = await call("open_session", net1=i["net1"], net2=i["net2"], spec=i["spec"], epsilon=i["epsilon"])
            await call("check", node=o["root"], rationale="root")
            st = await call("session_status")
            ok &= st["verdict"] == "UNSAFE"
            cert = str(work / "unsafe.json")
            await call("export_certificate", path=cert)
            v = run_checker(cert, i)
            print("checker:", v)
            ok &= v.startswith("ACCEPT UNSAFE")
    print("E2E", "OK" if ok else "FAILED", "workdir:", work)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main("--sandbox" in sys.argv)))
