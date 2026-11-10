"""Hardware inspection: per-GPU VRAM/PID accounting and host RAM."""

from __future__ import annotations

import shutil
import subprocess

import psutil

from vramctl.config import MIB, GPUInfo, SystemRAM

NVIDIA_SMI = "nvidia-smi"

_GPU_FIELDS = "index,uuid,name,memory.total,memory.used,utilization.gpu"
_APP_FIELDS = "pid,used_gpu_memory,gpu_uuid"


class NvidiaError(RuntimeError):
    """``nvidia-smi`` is missing, failed, or returned something unparseable."""


def nvidia_smi_available() -> bool:
    return shutil.which(NVIDIA_SMI) is not None


class LocalRunner:
    """Runs probes on this machine."""

    name = "local"
    remote = False

    def run(self, args: list[str], timeout: float) -> str:
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout, check=False
            )
        except FileNotFoundError as exc:
            raise NvidiaError(f"{args[0]} not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise NvidiaError(f"{args[0]} timed out after {timeout}s") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            raise NvidiaError(
                f"{args[0]} failed: {detail[0] if detail else 'unknown error'}"
            )
        return proc.stdout


class SshRunner:
    """Runs probes on another machine over SSH.

    ``BatchMode=yes`` keeps a missing key from hanging on a password prompt,
    which would otherwise stall a refresh until the timeout.
    """

    remote = True

    def __init__(self, destination: str, timeout: float = 4.0) -> None:
        self.destination = destination
        self.name = destination
        self.connect_timeout = max(1, int(timeout))

    def run(self, args: list[str], timeout: float) -> str:
        ssh_args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            self.destination,
            "--",
            *args,
        ]
        try:
            proc = subprocess.run(
                ssh_args, capture_output=True, text=True, timeout=timeout, check=False
            )
        except FileNotFoundError as exc:
            raise NvidiaError("ssh not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise NvidiaError(f"{self.destination}: unreachable after {timeout}s") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            first = detail[0] if detail else f"exit {proc.returncode}"
            raise NvidiaError(f"{self.destination}: {first}")
        return proc.stdout


def _run(args: list[str], timeout: float, runner=None) -> str:
    runner = runner or LocalRunner()
    return runner.run([NVIDIA_SMI, *args], timeout)


def _rows(output: str, expected: int) -> list[list[str]]:
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == expected:
            rows.append(cells)
    return rows


def _to_mb(value: str) -> int:
    """Parse ``nvidia-smi`` memory cells like ``322 MiB`` / ``322`` / ``[N/A]``."""
    token = value.replace("MiB", "").replace("MB", "").strip()
    try:
        return int(float(token))
    except ValueError:
        return 0


def _to_int(value: str) -> int:
    token = value.replace("%", "").strip()
    try:
        return int(float(token))
    except ValueError:
        return 0


def query_gpus(timeout: float = 5.0, runner=None, node: str = "") -> list[GPUInfo]:
    """Return every visible GPU with its VRAM totals and per-PID allocations.

    ``--query-compute-apps`` identifies GPUs by UUID rather than index, so the
    two queries are joined on UUID.
    """

    gpu_out = _run(
        ["--query-gpu=" + _GPU_FIELDS, "--format=csv,noheader,nounits"], timeout, runner
    )
    gpus: list[GPUInfo] = []
    by_uuid: dict[str, GPUInfo] = {}
    for index, uuid, name, total, used, util in _rows(gpu_out, 6):
        gpu = GPUInfo(
            index=_to_int(index),
            uuid=uuid,
            name=name,
            memory_total_mb=_to_mb(total),
            memory_used_mb=_to_mb(used),
            utilization_pct=_to_int(util),
            node=node,
        )
        gpus.append(gpu)
        by_uuid[uuid] = gpu

    try:
        app_out = _run(
            ["--query-compute-apps=" + _APP_FIELDS, "--format=csv,noheader,nounits"],
            timeout,
            runner,
        )
    except NvidiaError:
        # No compute apps (or an unsupported driver) leaves VRAM totals intact.
        return gpus

    for pid, used, uuid in _rows(app_out, 3):
        gpu = by_uuid.get(uuid)
        if gpu is None:
            continue
        try:
            pid_int = int(pid)
        except ValueError:
            continue
        gpu.pids[pid_int] = gpu.pids.get(pid_int, 0) + _to_mb(used)

    return gpus


def query_system_ram(runner=None, timeout: float = 5.0) -> SystemRAM:
    """Host RAM, from psutil locally or ``/proc/meminfo`` over SSH."""
    if runner is None or not getattr(runner, "remote", False):
        vm = psutil.virtual_memory()
        return SystemRAM(
            total_mb=vm.total // MIB,
            used_mb=(vm.total - vm.available) // MIB,
            available_mb=vm.available // MIB,
        )

    out = runner.run(["cat", "/proc/meminfo"], timeout)
    fields: dict[str, int] = {}
    for line in out.splitlines():
        key, _, rest = line.partition(":")
        value = rest.strip().split(" ")[0] if rest else ""
        try:
            fields[key.strip()] = int(value)  # kB
        except ValueError:
            continue
    total_kb = fields.get("MemTotal", 0)
    avail_kb = fields.get("MemAvailable", fields.get("MemFree", 0))
    return SystemRAM(
        total_mb=total_kb // 1024,
        used_mb=max(0, (total_kb - avail_kb) // 1024),
        available_mb=avail_kb // 1024,
    )


def declared_gpus(decls, node: str) -> list[GPUInfo]:
    """GPU rows from config, for a host that cannot be measured.

    ``memory_used_mb`` stays 0: nothing was measured, so the only honest
    number is what the backends themselves report holding.
    """
    return [
        GPUInfo(
            index=d.index,
            uuid=f"declared-{node}-{d.index}",
            name=d.name,
            memory_total_mb=d.vram_mb,
            memory_used_mb=0,
            utilization_pct=0,
            node=node,
            declared=True,
        )
        for d in decls
    ]


def gpu_pid_map(gpus: list[GPUInfo]) -> dict[int, dict[int, int]]:
    """Invert the GPU view: ``pid -> {gpu_index: MiB}``."""
    out: dict[int, dict[int, int]] = {}
    for gpu in gpus:
        for pid, mb in gpu.pids.items():
            out.setdefault(pid, {})[gpu.index] = mb
    return out


def process_ram_mb(pid: int) -> int:
    """Resident host memory for ``pid``, or 0 if it is gone or unreadable."""
    try:
        return psutil.Process(pid).memory_info().rss // MIB
    except (psutil.Error, OSError):
        return 0
