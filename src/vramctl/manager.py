"""State aggregation across machines and backends, plus the resplit controller."""

from __future__ import annotations

import time

from vramctl.backends.base import BackendError
from vramctl.backends.llamacpp import LlamaCppBackend
from vramctl.backends.ollama import OllamaBackend
from vramctl.backends.strata import StrataBackend
from vramctl.config import (
    AvailableModel,
    ClusterState,
    Config,
    GPUInfo,
    LoadedModel,
    NodeConfig,
    NodeState,
    PlacementConfig,
    load_config,
)
from vramctl.nvidia import (
    LocalRunner,
    NvidiaError,
    SshRunner,
    declared_gpus,
    query_gpus,
    query_system_ram,
)


class _Telemetry:
    """Cached GPU/RAM reading for one node.

    A local probe is cheap enough to repeat every tick. SSH is not: a round
    trip costs hundreds of milliseconds and the refresh runs on a timer, so a
    remote node's hardware is re-read on its own slower schedule and served
    from cache in between.
    """

    def __init__(self, node: NodeConfig, config: Config) -> None:
        self.node = node
        self.config = config
        self.runner = (
            LocalRunner()
            if node.local
            else (SshRunner(node.ssh, config.ssh_timeout) if node.ssh else None)
        )
        self.state = NodeState(name=node.name)
        self.fetched_at = 0.0

    @property
    def interval(self) -> float:
        if self.runner is None:
            return float("inf")
        return 0.0 if not self.runner.remote else self.config.remote_telemetry_seconds

    def read(self, force: bool = False) -> NodeState:
        if self.runner is None:
            # Nothing to measure: declared capacity is all we have.
            self.state = NodeState(
                name=self.node.name,
                gpus=declared_gpus(self.node.gpus, self.node.name),
                reachable=False,
                error="not measurable (no local access or ssh destination)",
            )
            return self.state

        age = time.time() - self.fetched_at
        if not force and self.fetched_at and age < self.interval:
            return self.state

        timeout = self.config.ssh_timeout if self.runner.remote else 5.0
        state = NodeState(name=self.node.name)
        try:
            state.gpus = query_gpus(timeout, self.runner, self.node.name)
            state.reachable = True
        except NvidiaError as exc:
            # Fall back to what config declares, flagged as unmeasured, so the
            # panel still shows the machine instead of dropping it.
            state.gpus = declared_gpus(self.node.gpus, self.node.name)
            state.reachable = False
            state.error = str(exc)

        try:
            state.ram = query_system_ram(self.runner, timeout)
        except (NvidiaError, OSError) as exc:
            if not state.error:
                state.error = f"ram: {exc}"

        self.state = state
        self.fetched_at = time.time()
        return state


class ClusterManager:
    """Single source of truth for the TUI.

    Hardware is read per node first so backends can attribute memory to real
    cards; a backend or host that is down becomes an entry in
    ``state.errors`` instead of taking the whole refresh with it.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.state = ClusterState()
        self.last_refresh = 0.0

        self._telemetry: dict[str, _Telemetry] = {}
        # (node, backend key) -> backend instance
        self.ollama: list[OllamaBackend] = []
        self.llamacpp: dict[str, LlamaCppBackend] = {}
        self.strata: dict[str, StrataBackend] = {}

        for node in self.config.nodes:
            self._telemetry[node.name] = _Telemetry(node, self.config)
            for endpoint in node.endpoints:
                self.ollama.append(OllamaBackend(self.config, endpoint, node.name))
            # llama.cpp and Strata read /proc, so they only work where we run.
            if node.llamacpp and node.local:
                self.llamacpp[node.name] = LlamaCppBackend(self.config)
            if node.strata_url and node.local:
                cfg = _with_strata_url(self.config, node.strata_url)
                self.strata[node.name] = StrataBackend(cfg)

    @property
    def nodes(self) -> list[NodeConfig]:
        return self.config.nodes

    def endpoints_for(self, node: str) -> list[OllamaBackend]:
        return [b for b in self.ollama if b.node == node]

    # --- read -----------------------------------------------------------

    def refresh_state(self, force_telemetry: bool = False) -> ClusterState:
        state = ClusterState()

        for node in self.config.nodes:
            node_state = self._telemetry[node.name].read(force=force_telemetry)
            state.nodes.append(node_state)
            if node_state.error:
                state.errors.append(f"{node.name}: {node_state.error}")

        by_node = {n.name: n for n in state.nodes}

        for backend in self.ollama:
            if not backend.is_enabled():
                continue
            node_state = by_node.get(backend.node)
            gpus = node_state.gpus if node_state else []
            try:
                state.models.extend(backend.list_running(gpus))
            except BackendError as exc:
                state.errors.append(str(exc))

        seen_pids: set[tuple[str, int]] = set()
        for name, process_backends in (
            ("llamacpp", self.llamacpp),
            ("strata", self.strata),
        ):
            for node_name, backend in process_backends.items():
                node_state = by_node.get(node_name)
                gpus = node_state.gpus if node_state else []
                try:
                    found = backend.list_running(gpus)
                except BackendError as exc:
                    state.errors.append(str(exc))
                    continue
                for model in found:
                    # One process must not appear under two backends.
                    if model.pid is not None:
                        token = (node_name, model.pid)
                        if token in seen_pids:
                            continue
                        seen_pids.add(token)
                    model.node = node_name
                    state.models.append(model)

        state.models.sort(key=lambda m: (m.node, -m.vram_mb, m.backend, m.name))
        self.state = state
        self.last_refresh = time.time()
        return state

    def list_available(self) -> list[AvailableModel]:
        """Everything loadable, per endpoint; unreachable backends are skipped."""
        out: list[AvailableModel] = []
        for backend in self.ollama:
            if not backend.is_enabled():
                continue
            try:
                for model in backend.list_available():
                    model.node = backend.node
                    model.endpoint = backend.label
                    model.pinned_gpu = backend.pinned_gpu
                    out.append(model)
            except BackendError:
                continue
        for mapping in (self.llamacpp, self.strata):
            for node_name, backend in mapping.items():
                try:
                    for model in backend.list_available():
                        model.node = node_name
                        out.append(model)
                except BackendError:
                    continue
        return sorted(out, key=lambda m: (m.backend, m.name))

    def gpu_for(self, node: str, index: int) -> GPUInfo | None:
        for node_state in self.state.nodes:
            if node_state.name != node:
                continue
            for gpu in node_state.gpus:
                if gpu.index == index:
                    return gpu
        return None

    # --- write ----------------------------------------------------------

    def _backend_for_model(self, model: LoadedModel):
        if model.backend == "ollama":
            match = next(
                (
                    b
                    for b in self.ollama
                    if b.node == model.node and b.label == model.endpoint
                ),
                None,
            )
            if match is None:
                raise BackendError(
                    f"no Ollama endpoint for {model.name} on {model.node}"
                )
            return match
        mapping = self.llamacpp if model.backend == "llamacpp" else self.strata
        backend = mapping.get(model.node)
        if backend is None:
            raise BackendError(f"{model.backend} is not managed on {model.node}")
        return backend

    def _backend_for_available(self, model: AvailableModel):
        """Resolve the endpoint-qualified handle produced by list_available."""
        if model.backend != "ollama":
            mapping = self.llamacpp if model.backend == "llamacpp" else self.strata
            backend = mapping.get(model.node) or next(iter(mapping.values()), None)
            if backend is None:
                raise BackendError(f"{model.backend} has no managed node")
            return backend, model

        match = next(
            (
                b
                for b in self.ollama
                if b.node == model.node and b.label == model.endpoint
            ),
            None,
        )
        if match is None:
            raise BackendError(
                f"no Ollama endpoint {model.endpoint!r} on {model.node!r}"
            )
        return match, model

    def load_model(self, model: AvailableModel, placement: PlacementConfig) -> None:
        backend, resolved = self._backend_for_available(model)
        backend.load(resolved, placement)
        self.refresh_state()

    def unload_model(self, model: LoadedModel) -> None:
        self._backend_for_model(model).unload(model)
        self.refresh_state()

    def resplit_model(
        self, model: LoadedModel, new_placement: PlacementConfig
    ) -> LoadedModel | None:
        """Move or re-split a loaded model: unload, then reload in place.

        None of these backends can migrate layers between devices in place, so
        this is a stop-and-reload. The unload is awaited before reloading,
        otherwise the old instance still holds the VRAM the new one needs.
        """

        backend = self._backend_for_model(model)
        available = AvailableModel(
            backend=model.backend,
            name=model.name,
            handle=model.handle,
            size_mb=model.size_mb,
            detail=model.detail,
        )

        backend.unload(model)
        self._await_release(model)

        target = backend
        if model.backend == "ollama" and new_placement.gpu_ids:
            # Moving between cards means moving between endpoints, since each
            # Ollama instance can only reach its own.
            wanted = new_placement.gpu_ids[0]
            match = next(
                (
                    b
                    for b in self.ollama
                    if b.node == model.node and b.pinned_gpu == wanted
                ),
                None,
            )
            if match is not None:
                target = match
        target.load(available, new_placement)

        state = self.refresh_state()
        return next(
            (
                m
                for m in state.models
                if m.backend == model.backend and m.handle == model.handle
            ),
            None,
        )

    def _await_release(self, model: LoadedModel, timeout: float = 30.0) -> None:
        """Block until the model's VRAM is actually back.

        Ollama's unload is asynchronous, and a killed process takes a moment
        for the driver to reclaim its allocation.
        """

        if model.vram_mb <= 0:
            return
        telemetry = self._telemetry.get(model.node)
        if telemetry is None:
            return
        deadline = time.time() + timeout
        baseline = model.vram_mb
        while time.time() < deadline:
            time.sleep(0.5)
            node_state = telemetry.read(force=True)
            if model.pid is not None:
                if not any(model.pid in gpu.pids for gpu in node_state.gpus):
                    return
                continue
            freed = baseline - sum(
                gpu.memory_used_mb
                for gpu in node_state.gpus
                if gpu.index in model.vram_mb_per_gpu
            )
            if freed >= 0:
                return
        return


def _with_strata_url(config: Config, url: str) -> Config:
    from dataclasses import replace

    return replace(config, strata_url=url.rstrip("/"))
