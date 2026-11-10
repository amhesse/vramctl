"""State aggregation across hardware and backends, plus the re-split controller."""

from __future__ import annotations

import time

from vramctl.backends.base import BackendError
from vramctl.backends.llamacpp import LlamaCppBackend
from vramctl.backends.ollama import OllamaBackend
from vramctl.backends.strata import StrataBackend
from vramctl.config import (
    LLAMACPP,
    OLLAMA,
    STRATA,
    AvailableModel,
    ClusterState,
    Config,
    LoadedModel,
    PlacementConfig,
    load_config,
)
from vramctl.nvidia import NvidiaError, query_gpus, query_system_ram


class ClusterManager:
    """Single source of truth for the TUI.

    Hardware is polled first so backends can attribute their memory to real
    GPUs; a backend that is down degrades to an entry in ``state.errors``
    instead of taking the whole refresh with it.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.ollama = OllamaBackend(self.config)
        self.llamacpp = LlamaCppBackend(self.config)
        self.strata = StrataBackend(self.config)
        self.state = ClusterState()
        self.last_refresh: float = 0.0

    @property
    def backends(self) -> dict[str, object]:
        return {OLLAMA: self.ollama, LLAMACPP: self.llamacpp, STRATA: self.strata}

    def backend_for(self, name: str):
        backend = self.backends.get(name)
        if backend is None:
            raise BackendError(f"unknown backend: {name}")
        return backend

    # --- read -----------------------------------------------------------

    def refresh_state(self) -> ClusterState:
        """Re-poll GPUs, RAM, and every enabled backend."""
        state = ClusterState()

        try:
            state.gpus = query_gpus()
        except NvidiaError as exc:
            state.errors.append(str(exc))

        try:
            state.ram = query_system_ram()
        except Exception as exc:  # psutil is platform-dependent
            state.errors.append(f"system RAM: {exc}")

        seen_pids: set[int] = set()
        for name, backend in self.backends.items():
            if not backend.is_enabled():
                continue
            try:
                models = backend.list_running(state.gpus)
            except BackendError as exc:
                state.errors.append(str(exc))
                continue
            for model in models:
                # A llama.cpp server spawned by Ollama would otherwise show up
                # under both backends; first claim wins.
                if model.pid is not None and model.pid in seen_pids:
                    continue
                if model.pid is not None:
                    seen_pids.add(model.pid)
                state.models.append(model)

        state.models.sort(key=lambda m: (-m.vram_mb, m.backend, m.name))
        self.state = state
        self.last_refresh = time.time()
        return state

    def list_available(self) -> list[AvailableModel]:
        """Everything loadable, across backends; failures are skipped quietly."""
        out: list[AvailableModel] = []
        for backend in self.backends.values():
            if not backend.is_enabled():
                continue
            try:
                out.extend(backend.list_available())
            except BackendError:
                continue
        return sorted(out, key=lambda m: (m.backend, m.name))

    def unallocated_vram_mb(self, gpu_index: int) -> int:
        """Free VRAM on a GPU, after what we know is already placed there."""
        for gpu in self.state.gpus:
            if gpu.index == gpu_index:
                return gpu.memory_free_mb
        return 0

    # --- write ----------------------------------------------------------

    def load_model(self, model: AvailableModel, placement: PlacementConfig) -> None:
        backend = self.backend_for(model.backend)
        backend.load(model, placement)
        self.refresh_state()

    def unload_model(self, model: LoadedModel) -> None:
        backend = self.backend_for(model.backend)
        backend.unload(model)
        self.refresh_state()

    def resplit_model(
        self, model: LoadedModel, new_placement: PlacementConfig
    ) -> LoadedModel | None:
        """Move or re-split a loaded model: unload, then reload in place.

        No backend here can migrate layers between devices in place, so this is
        a stop-and-reload. The unload is awaited before reloading, otherwise the
        old instance's VRAM is still held when the new one tries to allocate.
        """

        backend = self.backend_for(model.backend)
        available = AvailableModel(
            backend=model.backend,
            name=model.name,
            handle=model.handle,
            size_mb=model.size_mb,
            detail=model.detail,
        )

        backend.unload(model)
        self._await_release(model)
        backend.load(available, new_placement)

        state = self.refresh_state()
        return next(
            (m for m in state.models if m.backend == model.backend and m.handle == model.handle),
            None,
        )

    def _await_release(self, model: LoadedModel, timeout: float = 30.0) -> None:
        """Block until the model's VRAM is actually back.

        Ollama's unload is asynchronous and a killed process takes a moment to
        have its allocation reclaimed by the driver.
        """

        if model.vram_mb <= 0:
            return
        baseline = model.vram_mb
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                gpus = query_gpus()
            except NvidiaError:
                return
            if model.pid is not None:
                still_there = any(model.pid in gpu.pids for gpu in gpus)
                if not still_there:
                    return
            else:
                freed = sum(
                    self._prior_used(gpu.index) - gpu.memory_used_mb for gpu in gpus
                )
                if freed >= baseline * 0.8:
                    return
        return

    def _prior_used(self, gpu_index: int) -> int:
        for gpu in self.state.gpus:
            if gpu.index == gpu_index:
                return gpu.memory_used_mb
        return 0
