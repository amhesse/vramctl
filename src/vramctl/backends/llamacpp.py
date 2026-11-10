"""llama.cpp adapter: detect llama-server/llama-cli processes, spawn new ones."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import psutil

from vramctl.backends.base import BackendError
from vramctl.config import (
    LLAMACPP,
    AvailableModel,
    Config,
    GPUInfo,
    LoadedModel,
    PlacementConfig,
    ProcInfo,
    ProcessTable,
)


class LlamaCppBackend:
    """Owns ``llama-server`` subprocesses.

    llama.cpp exposes placement directly: ``CUDA_VISIBLE_DEVICES`` chooses the
    devices, ``--n-gpu-layers`` how much goes to VRAM, and ``--tensor-split``
    the per-device ratio. Note that ``-ts`` weights index into the *visible*
    devices, so they are ordered to match ``gpu_ids``.
    """

    name = LLAMACPP

    def __init__(self, config: Config, node: str = "", local: bool = True) -> None:
        self.config = config
        self.node = node
        # Detection works from any machine's process table; spawning and
        # stopping need the process to be ours, on this machine.
        self.local = local
        # pid -> (gguf path, placement) for servers we started ourselves
        self._spawned: dict[int, tuple[str, PlacementConfig]] = {}

    def is_enabled(self) -> bool:
        return self.config.llamacpp_enabled

    # --- inspection -----------------------------------------------------

    def list_running(
        self, gpus: list[GPUInfo], procs: ProcessTable | None = None
    ) -> list[LoadedModel]:
        return self.detect_processes(gpus, procs)

    def detect_processes(
        self, gpus: list[GPUInfo], procs: ProcessTable | None = None
    ) -> list[LoadedModel]:
        """Find llama.cpp processes in a machine's process table.

        A CPU-only ``llama-server`` holds no compute PID, so the whole table is
        scanned and the GPU map only supplies VRAM attribution.
        """

        if procs is None:
            from vramctl.probe import local_process_table

            procs = local_process_table(self.node)

        vram_by_pid: dict[int, dict[int, int]] = {}
        for gpu in gpus:
            for pid, mb in gpu.pids.items():
                vram_by_pid.setdefault(pid, {})[gpu.index] = mb

        models: list[LoadedModel] = []
        for proc in procs.matching(_is_llamacpp):
            gguf = _gguf_from_cmdline(proc.cmdline)
            if gguf is None and not procs.remote:
                # Only readable for a local process, and only useful when the
                # model came from a config file rather than the command line.
                gguf = _gguf_from_fds(proc.pid)
            per_gpu = vram_by_pid.get(proc.pid, {})
            placement = _placement_from_cmdline(proc.cmdline, sorted(per_gpu))
            # A blob path has no meaningful stem, so prefer the alias the
            # server was started with.
            label = (
                _alias_from_cmdline(proc.cmdline)
                or (Path(gguf).stem if gguf else None)
                or f"llama.cpp pid {proc.pid}"
            )

            models.append(
                LoadedModel(
                    backend=self.name,
                    name=label,
                    handle=gguf or str(proc.pid),
                    placement=placement,
                    pid=proc.pid,
                    vram_mb_per_gpu=per_gpu,
                    ram_mb=proc.rss_mb,
                    size_mb=_file_mb(gguf) if not procs.remote else 0,
                    detail=_detail(proc.cmdline, placement),
                    node=self.node,
                )
            )
        return models

    def discover_gguf_models(self, search_dirs: list[str] | None = None) -> list[AvailableModel]:
        """Walk the configured directories for ``.gguf`` files.

        Multi-part shards (``*-00002-of-00003.gguf``) collapse into their first
        part, which is the file llama.cpp is actually given.
        """

        dirs = search_dirs if search_dirs is not None else self.config.gguf_dirs
        found: dict[str, AvailableModel] = {}
        for raw in dirs:
            root = Path(raw).expanduser()
            if not root.is_dir():
                continue
            for path in root.rglob("*.gguf"):
                if not path.is_file():
                    continue
                stem = path.stem
                if "-of-" in stem:
                    shard = stem.rsplit("-", 3)
                    if len(shard) == 4 and shard[1] != "00001":
                        continue
                key = str(path)
                found[key] = AvailableModel(
                    backend=self.name,
                    name=path.stem,
                    handle=key,
                    size_mb=_file_mb(key),
                    detail=str(path.parent),
                )
        return sorted(found.values(), key=lambda m: m.name)

    def list_available(self) -> list[AvailableModel]:
        if not self.local:
            return []
        return self.discover_gguf_models()

    # --- control --------------------------------------------------------

    def load(self, model: AvailableModel, placement: PlacementConfig) -> None:
        self.spawn_model(
            gguf_path=model.handle,
            gpu_ids=placement.gpu_ids,
            n_gpu_layers=placement.n_gpu_layers,
            tensor_split=placement.tensor_split,
            port=placement.port,
        )

    def _require_local(self, action: str) -> None:
        if not self.local:
            raise BackendError(
                f"cannot {action} a llama.cpp server on {self.node}: vramctl "
                "only observes llama.cpp on remote machines"
            )

    def spawn_model(
        self,
        gguf_path: str,
        gpu_ids: list[int],
        n_gpu_layers: int | None = None,
        tensor_split: list[float] | None = None,
        port: int | None = None,
    ) -> int:
        """Start ``llama-server`` for ``gguf_path`` and return its pid."""

        self._require_local("start")
        binary = shutil.which(self.config.llama_server_bin)
        if binary is None:
            raise BackendError(
                f"{self.config.llama_server_bin} not found on PATH; set "
                "llama_server_bin in config or VRAMCTL_LLAMA_SERVER"
            )
        if not Path(gguf_path).is_file():
            raise BackendError(f"model file not found: {gguf_path}")

        chosen_port = port or self._free_port()
        layers = 0 if not gpu_ids else (999 if n_gpu_layers is None else n_gpu_layers)
        args = [
            binary,
            "--model", gguf_path,
            "--port", str(chosen_port),
            "--n-gpu-layers", str(layers),
        ]
        if tensor_split and len(gpu_ids) > 1:
            args += ["--tensor-split", ",".join(f"{w:g}" for w in tensor_split)]

        env = dict(os.environ)
        # -ts weights and device ids are relative to the visible set, so the
        # visible list is written in gpu_ids order.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids) if gpu_ids else ""

        try:
            proc = subprocess.Popen(
                args,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise BackendError(f"cannot start llama-server: {exc}") from exc

        # A model path llama.cpp rejects makes it exit within a second or two.
        time.sleep(1.0)
        if proc.poll() is not None:
            raise BackendError(
                f"llama-server exited immediately (code {proc.returncode}); "
                f"check {Path(gguf_path).name} and the layer split"
            )

        placement = PlacementConfig(
            gpu_ids=list(gpu_ids),
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            port=chosen_port,
        )
        self._spawned[proc.pid] = (gguf_path, placement)
        return proc.pid

    def unload(self, model: LoadedModel) -> None:
        if model.pid is None:
            raise BackendError(f"no pid recorded for {model.name}")
        self.stop_model(model.pid)

    def stop_model(self, pid: int, timeout: float = 10.0) -> None:
        """SIGTERM the server, escalating to SIGKILL so VRAM is actually freed."""
        self._require_local("stop")
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            self._spawned.pop(pid, None)
            return
        except psutil.Error as exc:
            raise BackendError(f"cannot signal pid {pid}: {exc}") from exc

        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except psutil.Error as exc:
                raise BackendError(f"pid {pid} would not die: {exc}") from exc
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, PermissionError) as exc:
            raise BackendError(f"not permitted to stop pid {pid}: {exc}") from exc
        finally:
            self._spawned.pop(pid, None)

    def supports_tensor_split(self) -> bool:
        return True

    def _free_port(self) -> int:
        base = self.config.llamacpp_base_port
        for offset in range(64):
            port = base + offset
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        raise BackendError(f"no free port in {base}-{base + 63}")


def _is_ollama_runner(proc: ProcInfo) -> bool:
    """True for the llama-server Ollama ships and runs as its own child.

    Ollama bundles a llama-server at a path like
    ``/usr/local/lib/ollama/llama-server`` and spawns it per loaded model.
    Those are already reported through ``/api/ps`` under a real tag, so
    claiming them here shows one model twice - once as ``codestral:22b`` and
    once as the sha256 of its blob.

    Only the binary's own path counts. A llama-server the user launched
    themselves stays visible even when it is pointed at an Ollama blob,
    because that is a process they manage and we should not hide it.
    """
    if not proc.cmdline:
        return False
    argv0 = proc.cmdline[0]
    return "/ollama/" in argv0 or Path(argv0).name == "ollama"


def _is_llamacpp(proc: ProcInfo) -> bool:
    """Match llama.cpp binaries without catching every ``server`` on the box."""
    if not proc.cmdline or _is_ollama_runner(proc):
        return False
    exe = proc.exe_name.lower()
    if exe in {"llama-server", "llama-cli", "llama-bench"}:
        return True
    # Locally built binaries are often just ./server or ./main, so require a
    # .gguf somewhere on the command line before claiming them.
    if exe in {"server", "main"}:
        return any(arg.endswith(".gguf") for arg in proc.cmdline)
    return False


def _alias_from_cmdline(cmdline: list[str]) -> str | None:
    for index, arg in enumerate(cmdline):
        if arg == "--alias" and index + 1 < len(cmdline):
            return cmdline[index + 1]
        if arg.startswith("--alias="):
            return arg.split("=", 1)[1]
    return None


def _gguf_from_cmdline(cmdline: list[str]) -> str | None:
    for index, arg in enumerate(cmdline):
        if arg in ("-m", "--model") and index + 1 < len(cmdline):
            return cmdline[index + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
        if arg.endswith(".gguf"):
            return arg
    return None


def _gguf_from_fds(pid: int) -> str | None:
    """Fall back to the open file descriptors for a model loaded via config."""
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.endswith(".gguf"):
            return target
    return None


def _placement_from_cmdline(cmdline: list[str], gpu_ids: list[int]) -> PlacementConfig:
    layers: int | None = None
    split: list[float] | None = None
    port: int | None = None

    for index, arg in enumerate(cmdline):
        nxt = cmdline[index + 1] if index + 1 < len(cmdline) else None
        if arg in ("-ngl", "--n-gpu-layers", "--gpu-layers") and nxt:
            layers = _int_or_none(nxt)
        elif arg.startswith("--n-gpu-layers="):
            layers = _int_or_none(arg.split("=", 1)[1])
        elif arg in ("-ts", "--tensor-split") and nxt:
            split = _floats(nxt)
        elif arg.startswith("--tensor-split="):
            split = _floats(arg.split("=", 1)[1])
        elif arg in ("--port",) and nxt:
            port = _int_or_none(nxt)
        elif arg.startswith("--port="):
            port = _int_or_none(arg.split("=", 1)[1])

    return PlacementConfig(
        gpu_ids=gpu_ids, n_gpu_layers=layers, tensor_split=split, port=port
    )


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _floats(value: str) -> list[float] | None:
    parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
    try:
        return [float(p) for p in parts] or None
    except ValueError:
        return None


def _file_mb(path: str | None) -> int:
    if not path:
        return 0
    try:
        return Path(path).stat().st_size // (1024 * 1024)
    except OSError:
        return 0


def _detail(cmdline: list[str], placement: PlacementConfig) -> str:
    bits = []
    if placement.port:
        bits.append(f":{placement.port}")
    if placement.n_gpu_layers is not None:
        bits.append(f"ngl={placement.n_gpu_layers}")
    if placement.tensor_split:
        bits.append("ts=" + "/".join(f"{w:g}" for w in placement.tensor_split))
    return " ".join(bits) or Path(cmdline[0]).name
