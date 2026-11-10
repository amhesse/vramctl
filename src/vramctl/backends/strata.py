"""Strata adapter: observe a Strata server and measure what it holds.

Strata is a local OpenAI/Anthropic-compatible server that holds **one model at
a time** and names it on ``/v1/status``:

    {"loaded": true, "model": "...", "service": "...", "engine": "...",
     "cache_max_tokens": 32768, "concurrency": {"serving": 1}}

That endpoint gives *identity*. It does not report memory, so the numbers come
from the hardware instead: the server's VRAM is read per GPU from
``nvidia-smi``'s compute-app table and its host memory from the process RSS,
joined on the pid that owns the listening port. Identity from the API,
measurement from the kernel - so the panel stays honest even if Strata's
reporting changes.

Strata's load/unload API is not verified, so this backend is observe-only
unless ``strata_load_path`` / ``strata_unload_path`` are configured.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import psutil

from vramctl.backends.base import BackendError
from vramctl.config import (
    STRATA,
    AvailableModel,
    Config,
    GPUInfo,
    LoadedModel,
    PlacementConfig,
    ProcInfo,
    ProcessTable,
)

MODEL_SUFFIXES = (".safetensors", ".gguf", ".strata")


class StrataBackend:
    name = STRATA

    def __init__(self, config: Config, node: str = "", local: bool = True) -> None:
        self.config = config
        self.base_url = config.strata_url
        # Status is polled on the refresh path, so it fails fast.
        self.timeout = config.inspect_timeout
        self.node = node
        self.local = local

    def is_enabled(self) -> bool:
        return self.config.strata_enabled

    # --- inspection -----------------------------------------------------

    def status(self) -> dict:
        """Raw ``/v1/status``. Raises BackendError if Strata is not there."""
        return self._get(self.config.strata_status_path)

    def list_running(
        self, gpus: list[GPUInfo], procs: ProcessTable | None = None
    ) -> list[LoadedModel]:
        return self.detect_models(gpus, procs)

    def detect_models(
        self, gpus: list[GPUInfo], procs: ProcessTable | None = None
    ) -> list[LoadedModel]:
        """The single model Strata currently holds, measured against hardware.

        Returns an empty list when Strata is unreachable or idle - an absent
        server is a normal state here, not an error worth surfacing.
        """

        if procs is None:
            from vramctl.probe import local_process_table

            procs = local_process_table(self.node)

        try:
            status = self.status()
        except BackendError:
            # Nothing answering. A Strata process may still be starting up, so
            # fall back to pure process detection rather than reporting nothing.
            return self._models_from_processes(gpus, procs)

        if not status.get("loaded", True):
            return []

        pid = self._server_pid(procs)
        per_gpu = self._vram_for_pid(gpus, pid)
        name = status.get("model") or "(unnamed)"

        detail_bits = [
            str(status.get("engine") or ""),
            _context_label(status),
            _concurrency_label(status),
        ]
        detail = " · ".join(b for b in detail_bits if b)

        return [
            LoadedModel(
                backend=self.name,
                name=str(name),
                handle=str(name),
                placement=PlacementConfig(gpu_ids=sorted(per_gpu)),
                pid=pid,
                vram_mb_per_gpu=per_gpu,
                ram_mb=procs.rss_mb(pid),
                size_mb=0,
                detail=detail or str(status.get("service") or "strata"),
                node=self.node,
            )
        ]

    def _models_from_processes(
        self, gpus: list[GPUInfo], procs: ProcessTable
    ) -> list[LoadedModel]:
        """Report a Strata process that holds GPU memory but answers nothing."""
        models: list[LoadedModel] = []
        for proc in procs.matching(_is_strata):
            per_gpu = self._vram_for_pid(gpus, proc.pid)
            if not per_gpu:
                # Idle and silent; nothing measurable to show.
                continue
            model_arg = next(
                (a for a in proc.cmdline if a.endswith(MODEL_SUFFIXES)), None
            )
            models.append(
                LoadedModel(
                    backend=self.name,
                    name=Path(model_arg).stem if model_arg else f"strata pid {proc.pid}",
                    handle=model_arg or str(proc.pid),
                    placement=PlacementConfig(gpu_ids=sorted(per_gpu)),
                    pid=proc.pid,
                    vram_mb_per_gpu=per_gpu,
                    ram_mb=proc.rss_mb,
                    size_mb=0,
                    detail="process-detected (status endpoint silent)",
                    node=self.node,
                )
            )
        return models

    def _vram_for_pid(self, gpus: list[GPUInfo], pid: int | None) -> dict[int, int]:
        if pid is None:
            return {}
        return {gpu.index: gpu.pids[pid] for gpu in gpus if pid in gpu.pids}

    def _server_pid(self, procs: ProcessTable) -> int | None:
        """The pid listening on Strata's port, falling back to a name scan.

        Matching on the listening port is exact, but the socket table is only
        readable locally, so a remote node falls back to the name scan.
        """
        port = _port_of(self.base_url)
        if port is not None and not procs.remote:
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if (
                        conn.laddr
                        and conn.laddr.port == port
                        and conn.status == psutil.CONN_LISTEN
                        and conn.pid
                    ):
                        return conn.pid
            except (psutil.AccessDenied, psutil.Error, OSError):
                pass
        matches = procs.matching(_is_strata)
        return matches[0].pid if matches else None

    def list_available(self) -> list[AvailableModel]:
        """What Strata could serve.

        Local only: the on-disk catalogue is read from this filesystem.

        Empty while the backend is observe-only: offering a model in the load
        modal that ``load()`` would then refuse is just noise. Configure
        ``strata_load_path`` and the on-disk catalogue appears.
        """
        if not self.config.strata_load_path or not self.local:
            return []
        return self.discover_strata_models()

    def discover_strata_models(
        self, search_dirs: list[str] | None = None
    ) -> list[AvailableModel]:
        dirs = search_dirs if search_dirs is not None else self.config.strata_dirs
        found: dict[str, AvailableModel] = {}
        for raw in dirs:
            root = Path(raw).expanduser()
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or not path.name.endswith(MODEL_SUFFIXES):
                    continue
                # Weights come in shards; the directory is the loadable unit.
                parent = path.parent
                if str(parent) in found:
                    continue
                found[str(parent)] = AvailableModel(
                    backend=self.name,
                    name=parent.name,
                    handle=str(parent),
                    size_mb=_dir_mb(parent),
                    detail=str(parent.parent),
                )
        return sorted(found.values(), key=lambda m: m.name)

    # --- control (opt-in) -----------------------------------------------

    def load(self, model: AvailableModel, placement: PlacementConfig) -> None:
        path = self.config.strata_load_path
        if not path:
            raise BackendError(
                "Strata is observe-only in this build: only /v1/status is a "
                "confirmed endpoint. Set strata_load_path in config to enable "
                "loading."
            )
        payload: dict[str, object] = {"model": model.handle}
        if placement.gpu_ids:
            payload["gpu_ids"] = list(placement.gpu_ids)
        if placement.ram_offload_mb is not None:
            payload["ram_offload_mb"] = placement.ram_offload_mb
        self._post(path, payload)

    def unload(self, model: LoadedModel) -> None:
        path = self.config.strata_unload_path
        if not path:
            raise BackendError(
                "Strata is observe-only in this build: only /v1/status is a "
                "confirmed endpoint. Set strata_unload_path in config, or stop "
                "the server yourself, to free its VRAM."
            )
        self._post(path, {"model": model.handle})

    def supports_tensor_split(self) -> bool:
        return False

    # --- transport ------------------------------------------------------

    def _get(self, path: str) -> dict:
        try:
            resp = httpx.get(f"{self.base_url}{path}", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"strata {path}: {exc}") from exc
        except ValueError as exc:
            raise BackendError(f"strata {path}: invalid JSON response") from exc
        if not isinstance(data, dict):
            raise BackendError(f"strata {path}: expected a JSON object")
        return data

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendError(f"strata {path}: {exc}") from exc
        except ValueError:
            return {}


def _is_strata(proc: ProcInfo) -> bool:
    return "strata" in proc.exe_name.lower()


def _port_of(base_url: str) -> int | None:
    from urllib.parse import urlparse

    try:
        return urlparse(base_url).port
    except ValueError:
        return None


def _context_label(status: dict) -> str:
    ctx = status.get("cache_max_tokens") or status.get("max_context")
    if not isinstance(ctx, int) or ctx <= 0:
        return ""
    return f"{ctx // 1024}k ctx" if ctx >= 1024 else f"{ctx} ctx"


def _concurrency_label(status: dict) -> str:
    serving = status.get("concurrency")
    if isinstance(serving, dict):
        serving = serving.get("serving")
    if not isinstance(serving, int):
        return ""
    return f"serving {serving}"


def _dir_mb(path: Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return 0
    return total // (1024 * 1024)
