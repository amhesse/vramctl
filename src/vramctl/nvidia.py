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


def _run(args: list[str], timeout: float) -> str:
    try:
        proc = subprocess.run(
            [NVIDIA_SMI, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise NvidiaError("nvidia-smi not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise NvidiaError(f"nvidia-smi timed out after {timeout}s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise NvidiaError(f"nvidia-smi failed: {detail[0] if detail else 'unknown error'}")
    return proc.stdout


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


def query_gpus(timeout: float = 5.0) -> list[GPUInfo]:
    """Return every visible GPU with its VRAM totals and per-PID allocations.

    ``--query-compute-apps`` identifies GPUs by UUID rather than index, so the
    two queries are joined on UUID.
    """

    gpu_out = _run(
        ["--query-gpu=" + _GPU_FIELDS, "--format=csv,noheader,nounits"], timeout
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
        )
        gpus.append(gpu)
        by_uuid[uuid] = gpu

    try:
        app_out = _run(
            ["--query-compute-apps=" + _APP_FIELDS, "--format=csv,noheader,nounits"],
            timeout,
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


def query_system_ram() -> SystemRAM:
    vm = psutil.virtual_memory()
    return SystemRAM(
        total_mb=vm.total // MIB,
        used_mb=(vm.total - vm.available) // MIB,
        available_mb=vm.available // MIB,
    )


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
