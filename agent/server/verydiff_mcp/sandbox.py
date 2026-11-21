"""Persistent exploration sandbox (Docker, no network).

Nothing produced here is ever read by the kernel. The only way information flows back is
through the agent's next proof action, which the kernel checks like any other.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath

DEFAULT_IMAGE = os.environ.get("VDA_SANDBOX_IMAGE", "verydiff-agent-sandbox:latest")
MAX_OUTPUT = int(os.environ.get("VDA_SANDBOX_MAX_OUTPUT", "8000"))


def _truncate(s: str, limit: int = MAX_OUTPUT) -> str:
    if len(s) <= limit:
        return s
    head = s[: limit // 2]
    tail = s[-limit // 2:]
    return f"{head}\n... [{len(s) - limit} chars truncated] ...\n{tail}"


class SandboxError(Exception):
    pass


class Sandbox:
    """One container per session. Host directories:
        <workdir>/nets     -> /nets     (read-only copies of the networks/spec)
        <workdir>/exports  -> /exports  (read-only; written by the kernel via export tools)
        <workdir>/scratch  -> /scratch  (read-write)
    """

    def __init__(self, workdir: str | os.PathLike, image: str = DEFAULT_IMAGE, cpus: str = "2",
                 memory: str = "4g", pids: int = 256, docker: str = "docker"):
        self.workdir = Path(workdir).resolve()
        self.nets = self.workdir / "nets"
        self.exports = self.workdir / "exports"
        self.scratch = self.workdir / "scratch"
        for d in (self.nets, self.exports, self.scratch):
            d.mkdir(parents=True, exist_ok=True)
        os.chmod(self.scratch, 0o777)
        self.image = image
        self.cpus, self.memory, self.pids = cpus, memory, pids
        self.docker = docker
        self.name: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def _run(self, args, timeout=60, check=True):
        r = subprocess.run([self.docker, *args], capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise SandboxError(f"docker {' '.join(args[:2])} failed: {r.stderr.strip()[:500]}")
        return r

    def start(self) -> str:
        if self.name and self.running():
            return self.name
        self.name = f"vdagent-{uuid.uuid4().hex[:10]}"
        self._run([
            "run", "-d", "--name", self.name,
            "--network", "none",
            "--cpus", self.cpus, "--memory", self.memory, "--pids-limit", str(self.pids),
            "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--tmpfs", "/tmp:rw,size=512m", "--tmpfs", "/home/agent:rw,size=256m,uid=1000,gid=1000",
            "-e", "JULIA_DEPOT_PATH=/tmp/julia-depot:/opt/julia-depot",
            "-v", f"{self.nets}:/nets:ro",
            "-v", f"{self.exports}:/exports:ro",
            "-v", f"{self.scratch}:/scratch:rw",
            "-w", "/scratch",
            self.image, "sleep", "infinity",
        ], timeout=120)
        return self.name

    def running(self) -> bool:
        if not self.name:
            return False
        r = self._run(["inspect", "-f", "{{.State.Running}}", self.name], check=False)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def stop(self) -> None:
        if self.name:
            self._run(["rm", "-f", self.name], check=False, timeout=60)
            self.name = None

    def reset(self) -> dict:
        """Fresh container; /scratch is wiped. /nets and /exports are kept."""
        self.stop()
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.scratch.mkdir(parents=True)
        os.chmod(self.scratch, 0o777)
        self.start()
        return {"container": self.name, "reset": True}

    # ------------------------------------------------------------------ tools
    def exec(self, cmd: str, timeout_s: int = 120) -> dict:
        self.start()
        timeout_s = max(1, min(int(timeout_s), 3600))
        try:
            r = subprocess.run([self.docker, "exec", self.name, "timeout", "-k", "5", str(timeout_s),
                                "bash", "-c", cmd], capture_output=True, text=True, timeout=timeout_s + 30)
            code, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            code, out, err = 124, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "host-side timeout"
        return {"exit_code": code, "stdout": _truncate(out), "stderr": _truncate(err, MAX_OUTPUT // 2),
                "timed_out": code == 124}

    def write_file(self, path: str, content: str) -> dict:
        """Write under /scratch only (resolved on the host bind mount; no traversal)."""
        p = PurePosixPath(path)
        if not p.is_absolute():
            p = PurePosixPath("/scratch") / p
        parts = []
        for part in p.parts[1:]:
            if part in ("", "."):
                continue
            if part == "..":
                raise SandboxError("path traversal is not allowed")
            parts.append(part)
        if not parts or parts[0] != "scratch" or len(parts) < 2:
            raise SandboxError("files can only be written under /scratch/")
        host = self.scratch.joinpath(*parts[1:])
        if not str(host.resolve()).startswith(str(self.scratch.resolve())):
            raise SandboxError("path escapes /scratch")
        host.parent.mkdir(parents=True, exist_ok=True)
        if host.is_symlink():
            raise SandboxError("refusing to write through a symlink")
        host.write_text(content)
        os.chmod(host, 0o666)
        return {"path": "/" + "/".join(parts), "bytes": len(content.encode())}

    def stage_inputs(self, files: dict[str, str]) -> dict:
        """Copy networks/spec to /nets (read-only inside the container)."""
        out = {}
        for name, src in files.items():
            dst = self.nets / name
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o644)
            out[name] = f"/nets/{name}"
        return out

    def export_path(self, name: str) -> tuple[Path, str]:
        safe = "".join(c for c in name if c.isalnum() or c in "-_.")
        return self.exports / safe, f"/exports/{safe}"
