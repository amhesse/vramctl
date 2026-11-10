"""Shared backend contract."""

from __future__ import annotations

from typing import Protocol

from vramctl.config import AvailableModel, GPUInfo, LoadedModel, PlacementConfig


class BackendError(RuntimeError):
    """A backend was unreachable, or refused a load/unload request."""


class Backend(Protocol):
    """What ``ClusterManager`` needs from every backend.

    ``list_running`` receives the live GPU list so backends that identify
    models by OS process (llama.cpp, Strata) can attribute VRAM per GPU.
    """

    name: str

    def is_enabled(self) -> bool: ...

    def list_running(self, gpus: list[GPUInfo]) -> list[LoadedModel]: ...

    def list_available(self) -> list[AvailableModel]: ...

    def load(self, model: AvailableModel, placement: PlacementConfig) -> None: ...

    def unload(self, model: LoadedModel) -> None: ...
