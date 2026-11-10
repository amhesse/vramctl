"""Core data types and configuration loading for vramctl."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

MIB = 1024 * 1024

OLLAMA = "ollama"
LLAMACPP = "llamacpp"
STRATA = "strata"


@dataclass
class GPUInfo:
    """A single NVIDIA GPU as reported by ``nvidia-smi``."""

    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int = 0
    # pid -> MiB that pid has allocated on *this* GPU
    pids: dict[int, int] = field(default_factory=dict)

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)

    @property
    def used_fraction(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return self.memory_used_mb / self.memory_total_mb

    @property
    def short_name(self) -> str:
        """``NVIDIA GeForce RTX 4070 Ti SUPER`` -> ``RTX 4070 Ti SUPER``."""
        name = self.name.removeprefix("NVIDIA ").removeprefix("GeForce ")
        return name.strip() or self.name


@dataclass
class SystemRAM:
    total_mb: int
    used_mb: int
    available_mb: int

    @property
    def used_fraction(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return self.used_mb / self.total_mb


@dataclass
class PlacementConfig:
    """Where a model should live, in backend-neutral terms.

    ``n_gpu_layers`` of ``None`` means "offload everything the backend can"
    (Ollama's default / llama.cpp's ``-ngl 999``). ``tensor_split`` holds one
    weight per entry in ``gpu_ids``; ``None`` lets the backend split evenly.
    """

    gpu_ids: list[int] = field(default_factory=list)
    n_gpu_layers: int | None = None
    tensor_split: list[float] | None = None
    ram_offload_mb: int | None = None
    port: int | None = None

    @property
    def main_gpu(self) -> int:
        return self.gpu_ids[0] if self.gpu_ids else 0

    @property
    def cpu_only(self) -> bool:
        return not self.gpu_ids or self.n_gpu_layers == 0

    def describe(self) -> str:
        if self.cpu_only:
            return "RAM only"
        gpus = "+".join(f"GPU{i}" for i in self.gpu_ids)
        layers = "all layers" if self.n_gpu_layers is None else f"{self.n_gpu_layers} layers"
        if self.tensor_split and len(self.gpu_ids) > 1:
            split = "/".join(f"{w:g}" for w in self.tensor_split)
            return f"{gpus} ({layers}, split {split})"
        return f"{gpus} ({layers})"

    def with_gpus(self, gpu_ids: list[int]) -> PlacementConfig:
        return replace(self, gpu_ids=list(gpu_ids))


@dataclass
class AvailableModel:
    """A model that *could* be loaded, as offered by a backend."""

    backend: str
    name: str
    # Opaque backend handle: an Ollama tag, a .gguf path, a Strata model id.
    handle: str
    size_mb: int = 0
    detail: str = ""


@dataclass
class LoadedModel:
    """A model currently resident in VRAM and/or host RAM."""

    backend: str
    name: str
    handle: str
    placement: PlacementConfig
    pid: int | None = None
    # gpu index -> MiB resident on that GPU
    vram_mb_per_gpu: dict[int, int] = field(default_factory=dict)
    ram_mb: int = 0
    size_mb: int = 0
    detail: str = ""

    @property
    def key(self) -> str:
        return f"{self.backend}:{self.handle}:{self.pid or 0}"

    @property
    def vram_mb(self) -> int:
        return sum(self.vram_mb_per_gpu.values())

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self.vram_mb_per_gpu) or list(self.placement.gpu_ids)


@dataclass
class ClusterState:
    gpus: list[GPUInfo] = field(default_factory=list)
    ram: SystemRAM | None = None
    models: list[LoadedModel] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def models_on_gpu(self, index: int) -> list[LoadedModel]:
        return [m for m in self.models if index in m.vram_mb_per_gpu]


@dataclass
class Config:
    ollama_url: str = "http://localhost:11434"
    ollama_enabled: bool = True

    llamacpp_enabled: bool = True
    llama_server_bin: str = "llama-server"
    llamacpp_base_port: int = 8080
    gguf_dirs: list[str] = field(default_factory=list)

    strata_enabled: bool = True
    # Strata serves one model at a time and names it on /v1/status. The port
    # is not yet confirmed for this host, so set it before trusting the panel.
    strata_url: str = "http://127.0.0.1:8181"
    strata_status_path: str = "/v1/status"
    # Strata's load/unload API is unverified; leave these unset and the backend
    # stays observe-only rather than POSTing to a guessed path.
    strata_load_path: str | None = None
    strata_unload_path: str | None = None
    strata_bin: str = "strata"
    strata_dirs: list[str] = field(default_factory=list)

    refresh_seconds: float = 3.0
    request_timeout: float = 10.0


DEFAULT_GGUF_DIRS = [
    "~/models",
    "~/.cache/huggingface/hub",
    "~/.local/share/models",
]

CONFIG_PATHS = [
    "~/.config/vramctl/config.json",
    "~/.vramctl.json",
]


def _expand_dirs(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        path = str(Path(value).expanduser())
        if path not in seen:
            seen.append(path)
    return seen


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from JSON, then let ``VRAMCTL_*`` env vars win.

    A missing config file is not an error; defaults target a local Ollama and
    the usual model cache directories.
    """

    cfg = Config(gguf_dirs=list(DEFAULT_GGUF_DIRS), strata_dirs=list(DEFAULT_GGUF_DIRS))

    candidates = [path] if path else CONFIG_PATHS
    for candidate in candidates:
        if candidate is None:
            continue
        file = Path(candidate).expanduser()
        if not file.is_file():
            continue
        try:
            raw = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read config {file}: {exc}") from exc
        known = {f.name for f in cfg.__dataclass_fields__.values()}
        for key, value in raw.items():
            if key in known:
                setattr(cfg, key, value)
        break

    env_map = {
        "VRAMCTL_OLLAMA_URL": "ollama_url",
        "VRAMCTL_STRATA_URL": "strata_url",
        "VRAMCTL_LLAMA_SERVER": "llama_server_bin",
        "VRAMCTL_STRATA_BIN": "strata_bin",
    }
    for env, attr in env_map.items():
        value = os.environ.get(env)
        if value:
            setattr(cfg, attr, value)

    if dirs := os.environ.get("VRAMCTL_GGUF_DIRS"):
        cfg.gguf_dirs = [d for d in dirs.split(os.pathsep) if d]

    cfg.ollama_url = cfg.ollama_url.rstrip("/")
    cfg.strata_url = cfg.strata_url.rstrip("/")
    cfg.gguf_dirs = _expand_dirs(cfg.gguf_dirs)
    cfg.strata_dirs = _expand_dirs(cfg.strata_dirs)
    return cfg
