"""Reading one machine's hardware and process table.

A local node is cheap to inspect: psutil and two ``nvidia-smi`` calls. A
remote node is not, so everything it needs is gathered in a *single* SSH
invocation whose sections are split on a sentinel. That keeps a refresh to one
round trip instead of four.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psutil

from vramctl.config import (
    GPUInfo,
    ProcessTable,
    ProcInfo,
    SystemRAM,
)
from vramctl.nvidia import (
    _APP_FIELDS,
    _GPU_FIELDS,
    NVIDIA_SMI,
    NvidiaError,
    _rows,
    _to_int,
    _to_mb,
    query_gpus,
    query_system_ram,
)

SENTINEL = "@@@vramctl@@@"

# One shell line per section, in the order parsed below. nvidia-smi failing
# here must not fail the whole probe, so each section is allowed to come back
# empty and is handled as "unmeasured" downstream.
_REMOTE_SCRIPT = (
    f"{NVIDIA_SMI} --query-gpu={_GPU_FIELDS} --format=csv,noheader,nounits 2>/dev/null; "
    f"echo {SENTINEL}; "
    f"{NVIDIA_SMI} --query-compute-apps={_APP_FIELDS} --format=csv,noheader,nounits 2>/dev/null; "
    f"echo {SENTINEL}; "
    "cat /proc/meminfo 2>/dev/null; "
    f"echo {SENTINEL}; "
    "ps -eo pid=,rss=,args= 2>/dev/null"
)


@dataclass
class NodeProbe:
    gpus: list[GPUInfo] = field(default_factory=list)
    ram: SystemRAM | None = None
    procs: ProcessTable = field(default_factory=ProcessTable)
    error: str = ""
    measured: bool = False


def probe_local(node: str) -> NodeProbe:
    probe = NodeProbe(procs=local_process_table(node))
    try:
        probe.gpus = query_gpus(5.0, None, node)
        probe.measured = True
    except NvidiaError as exc:
        probe.error = str(exc)
    try:
        probe.ram = query_system_ram()
    except Exception as exc:  # psutil is platform-dependent
        probe.error = probe.error or f"ram: {exc}"
    return probe


def probe_remote(runner, node: str, timeout: float) -> NodeProbe:
    """Everything one SSH call can tell us about a machine."""
    probe = NodeProbe()
    try:
        out = runner.run(["sh", "-c", _REMOTE_SCRIPT], timeout)
    except NvidiaError as exc:
        probe.error = str(exc)
        return probe

    sections = out.split(SENTINEL)
    while len(sections) < 4:
        sections.append("")
    gpu_out, app_out, meminfo, ps_out = sections[:4]

    probe.gpus = _parse_gpus(gpu_out, app_out, node)
    probe.measured = bool(probe.gpus)
    if not probe.measured:
        probe.error = "nvidia-smi returned nothing"
    probe.ram = _parse_meminfo(meminfo)
    probe.procs = _parse_ps(ps_out, node)
    return probe


def _parse_gpus(gpu_out: str, app_out: str, node: str) -> list[GPUInfo]:
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

    # Compute apps are identified by UUID, not index, so join on that.
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


def _parse_meminfo(text: str) -> SystemRAM | None:
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if not rest:
            continue
        try:
            fields[key.strip()] = int(rest.strip().split(" ")[0])  # kB
        except ValueError:
            continue
    total_kb = fields.get("MemTotal", 0)
    if not total_kb:
        return None
    avail_kb = fields.get("MemAvailable", fields.get("MemFree", 0))
    return SystemRAM(
        total_mb=total_kb // 1024,
        used_mb=max(0, (total_kb - avail_kb) // 1024),
        available_mb=avail_kb // 1024,
    )


def _parse_ps(text: str, node: str) -> ProcessTable:
    """Parse ``ps -eo pid=,rss=,args=`` output."""
    table = ProcessTable(node=node, remote=True)
    for line in text.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = int(parts[1])
        except ValueError:
            continue
        table.procs[pid] = ProcInfo(
            pid=pid, cmdline=parts[2].split(), rss_mb=rss_kb // 1024
        )
    return table


def local_process_table(node: str = "") -> ProcessTable:
    table = ProcessTable(node=node, remote=False)
    for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            pid = proc.info["pid"]
            cmdline = proc.info["cmdline"] or []
            mem = proc.info.get("memory_info")
        except (psutil.Error, OSError):
            continue
        if not cmdline:
            continue
        table.procs[pid] = ProcInfo(
            pid=pid,
            cmdline=list(cmdline),
            rss_mb=(mem.rss // (1024 * 1024)) if mem else 0,
        )
    return table
