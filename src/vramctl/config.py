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
    # Which machine this card is in, and whether the totals were measured by
    # nvidia-smi or merely declared in config because the host was unreachable.
    node: str = ""
    declared: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.node, self.index)

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
    # Where loading it would land. For a pinned Ollama endpoint the GPU is not
    # a choice: the endpoint is the card.
    node: str = ""
    endpoint: str = ""
    pinned_gpu: int | None = None


@dataclass
class LoadedModel:
    """A model currently resident in VRAM and/or host RAM."""

    backend: str
    name: str
    handle: str
    placement: PlacementConfig
    pid: int | None = None
    # gpu index -> MiB resident on that GPU (indices are local to ``node``)
    vram_mb_per_gpu: dict[int, int] = field(default_factory=dict)
    ram_mb: int = 0
    size_mb: int = 0
    detail: str = ""
    # Which machine holds this model, and the endpoint it was seen through.
    node: str = ""
    endpoint: str = ""

    @property
    def key(self) -> str:
        return f"{self.node}:{self.backend}:{self.endpoint}:{self.handle}:{self.pid or 0}"

    @property
    def spread(self) -> str:
        """How the model is split between VRAM and host RAM."""
        if self.vram_mb and self.ram_mb:
            return "split"
        if self.vram_mb:
            return "vram"
        return "ram"

    @property
    def vram_mb(self) -> int:
        return sum(self.vram_mb_per_gpu.values())

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self.vram_mb_per_gpu) or list(self.placement.gpu_ids)


@dataclass
class GPUDecl:
    """Declared GPU facts, used when a host cannot be measured."""

    index: int
    name: str = "GPU"
    vram_mb: int = 0


@dataclass
class EndpointConfig:
    """One backend endpoint, optionally pinned to a single GPU.

    ``gpu`` is the index *on its node*. When set, the endpoint is the GPU's
    identity: everything it reports lives on that card, and no PID guessing is
    needed. It also means the instance sees exactly one device, so a load must
    address it as device 0.
    """

    url: str
    gpu: int | None = None
    label: str = ""
    backend: str = OLLAMA

    @property
    def name(self) -> str:
        return self.label or self.url


@dataclass
class NodeConfig:
    """A machine holding GPUs.

    ``local`` nodes are measured with a subprocess; ``ssh`` nodes through that
    SSH destination. Process-detecting backends (llama.cpp, Strata) need
    ``/proc``, so they run on local nodes only.
    """

    name: str
    local: bool = False
    ssh: str | None = None
    endpoints: list[EndpointConfig] = field(default_factory=list)
    llamacpp: bool = False
    strata_url: str | None = None
    gpus: list[GPUDecl] = field(default_factory=list)

    @property
    def measurable(self) -> bool:
        return self.local or bool(self.ssh)


@dataclass
class NodeState:
    name: str
    gpus: list[GPUInfo] = field(default_factory=list)
    ram: SystemRAM | None = None
    reachable: bool = True
    error: str = ""


@dataclass
class ClusterState:
    nodes: list[NodeState] = field(default_factory=list)
    models: list[LoadedModel] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def gpus(self) -> list[GPUInfo]:
        """Every GPU in the cluster, flattened."""
        return [gpu for node in self.nodes for gpu in node.gpus]

    @property
    def ram(self) -> SystemRAM | None:
        """Host RAM of the first local node, for the single-machine case."""
        return next((n.ram for n in self.nodes if n.ram), None)

    def models_on_gpu(self, node: str, index: int) -> list[LoadedModel]:
        return [
            m
            for m in self.models
            if m.node == node and index in m.vram_mb_per_gpu
        ]

    def models_on_node(self, node: str) -> list[LoadedModel]:
        return [m for m in self.models if m.node == node]


@dataclass
class Config:
    # One entry per machine. Built from the legacy flat fields below when the
    # config file does not declare any.
    nodes: list[NodeConfig] = field(default_factory=list)

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
    # SSH is far slower than a local subprocess, so remote GPU telemetry is
    # cached between refreshes rather than re-polled on every tick.
    ssh_timeout: float = 4.0
    remote_telemetry_seconds: float = 15.0


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

    raw_nodes: list | None = None
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
        raw_nodes = raw.get("nodes")
        known = {f.name for f in cfg.__dataclass_fields__.values()}
        for key, value in raw.items():
            if key in known and key != "nodes":
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
    cfg.nodes = _build_nodes(cfg, raw_nodes)
    return cfg


def _build_nodes(cfg: Config, raw_nodes: list | None) -> list[NodeConfig]:
    """Parse ``nodes[]``, or synthesize one local node from the flat fields.

    Without a ``nodes`` key the tool behaves as it always did: this machine,
    one Ollama URL, plus the process-detecting backends.
    """

    if not raw_nodes:
        return [
            NodeConfig(
                name=_local_name(),
                local=True,
                endpoints=(
                    [EndpointConfig(url=cfg.ollama_url)] if cfg.ollama_enabled else []
                ),
                llamacpp=cfg.llamacpp_enabled,
                strata_url=cfg.strata_url if cfg.strata_enabled else None,
            )
        ]

    nodes: list[NodeConfig] = []
    for entry in raw_nodes:
        if not isinstance(entry, dict):
            raise ValueError(f"each node must be an object, got {type(entry).__name__}")
        ssh = entry.get("ssh")
        local = bool(entry.get("local", not ssh))
        name = str(entry.get("name") or ssh or (_local_name() if local else "node"))

        endpoints: list[EndpointConfig] = []
        for raw in entry.get("ollama", []) or []:
            if isinstance(raw, str):
                endpoints.append(EndpointConfig(url=raw.rstrip("/")))
                continue
            if not isinstance(raw, dict) or not raw.get("url"):
                raise ValueError(f"node {name}: each ollama entry needs a url")
            gpu = raw.get("gpu")
            endpoints.append(
                EndpointConfig(
                    url=str(raw["url"]).rstrip("/"),
                    gpu=int(gpu) if gpu is not None else None,
                    label=str(raw.get("label") or ""),
                )
            )

        gpus: list[GPUDecl] = []
        for raw in entry.get("gpus", []) or []:
            if not isinstance(raw, dict):
                continue
            gpus.append(
                GPUDecl(
                    index=int(raw.get("index", len(gpus))),
                    name=str(raw.get("name") or "GPU"),
                    vram_mb=int(raw.get("vram_mb") or 0),
                )
            )

        strata = entry.get("strata")
        strata_url = None
        if isinstance(strata, str):
            strata_url = strata.rstrip("/")
        elif isinstance(strata, dict) and strata.get("url"):
            strata_url = str(strata["url"]).rstrip("/")

        nodes.append(
            NodeConfig(
                name=name,
                local=local,
                ssh=str(ssh) if ssh else None,
                endpoints=endpoints,
                llamacpp=bool(entry.get("llamacpp", local)),
                strata_url=strata_url,
                gpus=gpus,
            )
        )
    return nodes


def _local_name() -> str:
    import socket

    try:
        return socket.gethostname() or "local"
    except OSError:
        return "local"
