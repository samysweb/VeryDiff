"""Client for the long-lived Julia kernel process (newline-delimited JSON-RPC over stdio).

The kernel holds ALL proof state; this client only forwards requests. It never interprets
results as evidence.
"""
from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = AGENT_DIR / "julia" / "VeryDiffAgentKernel"


class KernelError(Exception):
    """Structured error returned by the kernel (precondition failures etc.)."""

    def __init__(self, type_: str, message: str):
        super().__init__(f"{type_}: {message}")
        self.type = type_
        self.message = message


class KernelTimeout(Exception):
    pass


class KernelClient:
    def __init__(self, project: str | os.PathLike | None = None, julia: str | None = None,
                 sysimage: str | None = None, log_path: str | os.PathLike | None = None,
                 warmup: bool = True):
        self.project = str(project or os.environ.get("VDA_KERNEL_PROJECT", DEFAULT_PROJECT))
        self.julia = julia or os.environ.get("VDA_JULIA", "julia")
        self.sysimage = sysimage or os.environ.get("VDA_SYSIMAGE") or None
        self.log_path = str(log_path or os.environ.get("VDA_KERNEL_LOG", "/tmp/verydiff_kernel.log"))
        self.warmup_enabled = warmup and os.environ.get("VDA_WARMUP", "1") != "0"
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._start_lock = threading.RLock()
        self.ready = False
        self.startup_seconds: float | None = None
        self.warmup_seconds: float | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self, timeout: float = 600.0) -> None:
        with self._start_lock:
            self._start(timeout)

    def _start(self, timeout: float) -> None:
        if self.proc and self.proc.poll() is None and self.ready:
            return
        if self.proc and self.proc.poll() is None:
            self.proc.kill()      # half-started (e.g. failed warm-up): restart cleanly
        self.ready = False
        t0 = time.time()
        cmd = [self.julia, f"--project={self.project}", "--startup-file=no"]
        if self.sysimage:
            cmd.append(f"--sysimage={self.sysimage}")
        cmd += ["-e", "using VeryDiffAgentKernel; VeryDiffAgentKernel.serve()"]
        env = dict(os.environ)
        env.setdefault("VERYDIFF_NO_GUROBI", "1")
        logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logf,
                                     env=env, text=True, bufsize=1)
        self._lines = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        ready = self._next_line(timeout)
        if not ready or not json.loads(ready).get("result", {}).get("ready"):
            raise RuntimeError(f"kernel failed to start (see {self.log_path})")
        self.startup_seconds = time.time() - t0
        if self.warmup_enabled:
            r = self._rpc("warmup", None, timeout)
            self.warmup_seconds = r.get("warmup_seconds")
        self.ready = True

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _next_line(self, timeout: float) -> str | None:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise KernelTimeout(f"no response from kernel within {timeout}s")
        if line is None:
            raise RuntimeError(f"kernel process exited (see {self.log_path})")
        return line

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"id": 0, "method": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        self.proc = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------------ calls
    def call(self, method: str, params: dict | None = None, timeout: float | None = None):
        """Send one request and wait for its response. Responses to earlier timed-out
        requests (stale ids) are discarded."""
        timeout = timeout or float(os.environ.get("VDA_CALL_TIMEOUT", "900"))
        self.start()          # blocks while a (background) start + warm-up is in progress
        return self._rpc(method, params, timeout)

    def _rpc(self, method: str, params: dict | None, timeout: float):
        with self._lock:
            if not self.alive():
                raise RuntimeError(f"kernel process is not running (see {self.log_path})")
            rid = next(self._ids)
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise KernelTimeout(f"kernel call '{method}' exceeded {timeout}s (it keeps running; "
                                        f"later calls wait for it)")
                resp = json.loads(self._next_line(remaining))
                if resp.get("id") != rid:
                    continue  # stale response of a timed-out call
                if "error" in resp:
                    e = resp["error"]
                    raise KernelError(e.get("type", "internal"), e.get("message", ""))
                return resp["result"]
