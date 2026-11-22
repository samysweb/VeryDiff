#!/usr/bin/env python3
"""Headless benchmark: LLM agent (Claude Agent SDK) driving the verydiff MCP server vs plain VeryDiff CLI.

  python3 agent/harness/run_benchmark.py [--instances agent/tests/nets/instances.json]
         [--agent claude|scripted|none] [--model MODEL] [--timeout 600] [--max-turns 80]
         [--baseline] [--out agent/harness/runs/<timestamp>] [--only NAME ...]

--agent claude   : real agent through the Claude Agent SDK (needs Claude Code CLI + credentials).
--agent scripted : deterministic stand-in (check root, then auto_expand until budget) through the same
                   MCP server; validates the pipeline without an LLM.
--agent none     : baseline only.

Per instance: a fresh MCP server (and kernel), JSONL log of every tool call with timings, the certificate
(exported by the agent, else by the server on exit), the independent checker's verdict, and the VeryDiff
CLI baseline. Writes summary.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
REPO = AGENT.parent
KERNEL = AGENT / "julia" / "VeryDiffAgentKernel"
CHECKER = AGENT / "checker" / "check_certificate.jl"
PROMPT = AGENT / "harness" / "prompts" / "agent_system.md"


def load_instances(path: Path):
    base = path.parent
    out = []
    for i in json.loads(path.read_text()):
        d = dict(i)
        for k in ("net1", "net2", "spec"):
            d[k] = str((base / i[k]).resolve()) if not os.path.isabs(i[k]) else i[k]
        d.setdefault("timeout", None)
        out.append(d)
    return out


OBJECTIVES = {
    "verdict": "",
    "min-nodes": ("\nObjective: besides a correct verdict, MINIMISE the total number of proof-tree nodes "
                  "(session_status.total_nodes). auto_expand is capped at 100 nodes per call. Use the weight-space "
                  "reading workflow (read_weights, inspect_layer, split_scores, the sandbox) to choose splits and "
                  "lemmas that a generic bisection heuristic would not find, and justify each in its rationale."),
}
MAX_AUTO = {"verdict": None, "min-nodes": "100"}


def server_config(work: Path, cert: Path, objective: str = "verdict") -> dict:
    env = {
        "PYTHONPATH": str(AGENT / "server"),
        "VDA_WORKDIR": str(work),
        "VDA_FINAL_CERT": str(cert),
        "VERYDIFF_NO_GUROBI": "1",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    if MAX_AUTO.get(objective):
        env["VDA_MAX_AUTO_NODES"] = MAX_AUTO[objective]
    for k in ("VDA_SANDBOX", "VDA_SANDBOX_IMAGE", "VDA_SYSIMAGE", "VDA_JULIA", "JULIA_DEPOT_PATH"):
        if k in os.environ:
            env[k] = os.environ[k]
    return {"type": "stdio", "command": sys.executable, "args": ["-m", "verydiff_mcp.server"], "env": env}


def task_prompt(inst: dict, cert: Path, timeout: float, objective: str = "verdict") -> str:
    return (f"Decide ε-equivalence for:\n  net1 = {inst['net1']}\n  net2 = {inst['net2']}\n"
            f"  spec = {inst['spec']}\n  epsilon = {inst['epsilon']}\n"
            f"Wall-clock budget: {int(timeout)} s. When done (or near the budget), call "
            f"export_certificate with path=\"{cert}\"." + OBJECTIVES[objective])


# ----------------------------------------------------------------------------- agents
async def run_claude(inst, work: Path, cert: Path, log, timeout: float, max_turns: int, model: str | None,
                     objective: str = "verdict"):
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolResultBlock,
                                  ToolUseBlock, UserMessage, query)
    opts = ClaudeAgentOptions(
        system_prompt=PROMPT.read_text(),
        mcp_servers={"verydiff": server_config(work, cert, objective)},
        env={"MCP_TOOL_TIMEOUT": str(int((timeout + 120) * 1000)), "MCP_TIMEOUT": "600000"},
        strict_mcp_config=True,
        tools=[],                                   # no built-in tools: only the verydiff MCP tools
        allowed_tools=["mcp__verydiff"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        max_turns=max_turns,
        model=model,
        cwd=str(work),
    )
    pending: dict[str, tuple[str, float]] = {}
    stats = {"tool_calls": 0, "turns": 0, "cost_usd": None, "result": None}

    async def consume():
        async for msg in query(prompt=task_prompt(inst, cert, timeout, objective), options=opts):
            now = time.time()
            if isinstance(msg, AssistantMessage):
                stats["turns"] += 1
                for b in msg.content:
                    if isinstance(b, ToolUseBlock):
                        pending[b.id] = (b.name, now)
                        stats["tool_calls"] += 1
                        log({"t": now, "event": "tool_use", "id": b.id, "tool": b.name, "input": b.input})
            elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        name, t0 = pending.pop(b.tool_use_id, ("?", now))
                        text = b.content if isinstance(b.content, str) else json.dumps(b.content, default=str)
                        log({"t": now, "event": "tool_result", "id": b.tool_use_id, "tool": name,
                             "seconds": round(now - t0, 3), "is_error": b.is_error, "content": text[:4000]})
            elif isinstance(msg, ResultMessage):
                stats["cost_usd"] = msg.total_cost_usd
                stats["result"] = (msg.result or "")[:2000]
                log({"t": now, "event": "result", "stop_reason": msg.stop_reason, "cost_usd": msg.total_cost_usd,
                     "usage": msg.usage})

    try:
        await asyncio.wait_for(consume(), timeout=timeout)
    except asyncio.TimeoutError:
        log({"t": time.time(), "event": "wall_clock_budget_exhausted"})
    return stats


async def run_scripted(inst, work: Path, cert: Path, log, timeout: float, objective: str = "verdict", **_):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    cfg = server_config(work, cert, objective)
    params = StdioServerParameters(command=cfg["command"], args=cfg["args"], env=cfg["env"])
    stats = {"tool_calls": 0, "turns": 0, "cost_usd": 0.0, "result": None}
    t_end = time.time() + timeout
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            async def call(name, **args):
                t0 = time.time()
                log({"t": t0, "event": "tool_use", "tool": name, "input": args})
                res = await s.call_tool(name, args, read_timeout_seconds=max(60, t_end - time.time() + 120))
                sc = res.structured_content or {}
                out = sc.get("result", sc) if isinstance(sc, dict) else sc
                if not out:
                    out = json.loads("".join(getattr(c, "text", "") for c in res.content) or "{}")
                stats["tool_calls"] += 1
                log({"t": time.time(), "event": "tool_result", "tool": name, "seconds": round(time.time() - t0, 3),
                     "content": json.dumps(out)[:4000]})
                return out

            o = await call("open_session", net1=inst["net1"], net2=inst["net2"], spec=inst["spec"], epsilon=inst["epsilon"])
            root = o["root"]
            await call("check", node=root, rationale="scripted: root")
            while time.time() < t_end - 5:
                st = await call("session_status")
                if st["verdict"] != "PARTIAL":
                    break
                await call("auto_expand", node=root, max_nodes=2000, max_seconds=max(1.0, min(60.0, t_end - time.time() - 5)),
                           rationale="scripted: budgeted VeryDiff search")
            await call("export_certificate", path=str(cert))
    return stats


# ----------------------------------------------------------------------------- checker / baseline
def run_checker(cert: Path, inst) -> str:
    if not cert.exists():
        return "NO_CERTIFICATE"
    r = subprocess.run(["julia", f"--project={KERNEL}", str(CHECKER), str(cert), inst["net1"], inst["net2"], inst["spec"]],
                       capture_output=True, text=True, env={**os.environ, "VERYDIFF_NO_GUROBI": "1"})
    lines = [l for l in r.stdout.splitlines() if l.startswith(("ACCEPT", "REJECT"))]
    return lines[-1] if lines else f"CHECKER_FAILED: {r.stderr[-300:]}"


def run_baseline(inst, timeout: float) -> tuple[str, float]:
    t0 = time.time()
    r = subprocess.run(["julia", f"--project={REPO}", str(REPO / "cli.jl"), inst["net1"], inst["net2"], inst["spec"],
                        "--epsilon", str(inst["epsilon"]), "--timeout", str(int(timeout))],
                       capture_output=True, text=True, env={**os.environ, "VERYDIFF_NO_GUROBI": "1"},
                       timeout=timeout + 300)
    verdict = "UNKNOWN"
    for line in r.stdout.splitlines():
        if line.strip() in ("SAFE", "UNSAFE", "UNKNOWN"):
            verdict = line.strip()
    return verdict, time.time() - t0


def cert_stats(cert: Path) -> dict:
    if not cert.exists():
        return {}
    raw = cert.read_bytes()
    C = json.loads(gzip.decompress(raw) if cert.suffix == ".gz" else raw)
    proposed = sum(1 for e in C["log"] if e["action"] == "introduce_lemma")
    proven = sum(1 for l in C["lemmas"] if l["status"] == "proven")
    nodes = C["stats"]["nodes"]
    auto = C["stats"]["auto_nodes"]
    return {"status": C["status"], "nodes": nodes, "auto_nodes": auto, "manual_nodes": nodes - auto,
            "lemmas_proposed": proposed, "lemmas_proven": proven, "lemmas_used": len(C["lemmas"]),
            "proof_actions": len(C["log"]), "cert_bytes": len(raw), "kernel_seconds": C["stats"]["kernel_seconds"]}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=str(AGENT / "tests" / "nets" / "instances.json"))
    ap.add_argument("--agent", choices=["claude", "scripted", "none"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="verdict")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or AGENT / "harness" / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for inst in load_instances(Path(a.instances)):
        if a.only and inst["name"] not in a.only:
            continue
        timeout = inst.get("timeout") or a.timeout
        work = out / inst["name"]
        work.mkdir(exist_ok=True)
        cert = work / "certificate.json"
        row = {"instance": inst["name"], "epsilon": inst["epsilon"], "expected": inst.get("expected", "")}
        if a.agent != "none":
            logf = open(work / "calls.jsonl", "w")
            log = lambda d: (logf.write(json.dumps(d, default=str) + "\n"), logf.flush())
            t0 = time.time()
            runner = run_claude if a.agent == "claude" else run_scripted
            try:
                stats = asyncio.run(runner(inst, work, cert, log, timeout, max_turns=a.max_turns, model=a.model,
                                         objective=a.objective))
            except Exception as e:
                log({"event": "harness_error", "error": repr(e)})
                stats = {"tool_calls": None, "cost_usd": None}
            row["agent_wall_s"] = round(time.time() - t0, 1)
            row["llm_actions"] = stats.get("tool_calls")
            row["cost_usd"] = stats.get("cost_usd")
            logf.close()
            cs = cert_stats(cert)
            row["agent_verdict"] = cs.get("status", "NONE")
            row["checker"] = run_checker(cert, inst)
            row.update({k: v for k, v in cs.items() if k != "status"})
        if a.baseline:
            v, t = run_baseline(inst, timeout)
            row["baseline_verdict"], row["baseline_wall_s"] = v, round(t, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "instance", k))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("summary:", out / "summary.csv")


if __name__ == "__main__":
    main()
