"""Shared backend contract."""

from __future__ import annotations

from typing import Protocol

from vramctl.config import (
    AvailableModel,
    GPUInfo,
    LoadedModel,
    PlacementConfig,
    ProcessTable,
)


class BackendError(RuntimeError):
    """A backend was unreachable, or refused a load/unload request."""


class Backend(Protocol):
    """What ``ClusterManager`` needs from every backend.

    ``list_running`` receives the live GPU list so backends that identify
    models by OS process (llama.cpp, Strata) can attribute VRAM per GPU, plus
    that machine's process table - which may have been read over SSH, so no
    backend may assume ``/proc`` is reachable.
    """

    name: str

    def is_enabled(self) -> bool: ...

    def list_running(
        self, gpus: list[GPUInfo], procs: ProcessTable | None = None
    ) -> list[LoadedModel]: ...

    def list_available(self) -> list[AvailableModel]: ...

    def load(self, model: AvailableModel, placement: PlacementConfig) -> None: ...

    def unload(self, model: LoadedModel) -> None: ...
