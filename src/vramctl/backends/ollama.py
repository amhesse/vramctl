"""Ollama adapter: inspect /api/ps and /api/tags, place models via /api/generate."""

from __future__ import annotations

import httpx
import psutil

from vramctl.backends.base import BackendError
from vramctl.config import (
    MIB,
    OLLAMA,
    AvailableModel,
    Config,
    GPUInfo,
    LoadedModel,
    PlacementConfig,
)


class OllamaBackend:
    """Talks to the Ollama HTTP API.

    Ollama owns placement itself, so the knobs are coarse: ``num_gpu`` sets how
    many layers go to VRAM (0 = pure RAM) and ``main_gpu`` picks the device.
    There is no per-GPU tensor split, so a multi-GPU placement is passed as
    ``main_gpu`` plus a ``CUDA_VISIBLE_DEVICES`` hint Ollama only honours if it
    was set on the server — we surface that limit rather than pretend.
    """

    name = OLLAMA

    def __init__(self, config: Config) -> None:
        self.config = config
        self.base_url = config.ollama_url
        self.timeout = config.request_timeout

    def is_enabled(self) -> bool:
        return self.config.ollama_enabled

    # --- inspection -----------------------------------------------------

    def _get(self, path: str) -> dict:
        try:
            resp = httpx.get(f"{self.base_url}{path}", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama {path}: {exc}") from exc
        except ValueError as exc:
            raise BackendError(f"ollama {path}: invalid JSON response") from exc

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama {path}: {exc}") from exc
        except ValueError:
            return {}

    def list_running(self, gpus: list[GPUInfo]) -> list[LoadedModel]:
        """Map ``GET /api/ps`` onto ``LoadedModel``.

        Ollama reports ``size`` (total resident) and ``size_vram`` (the VRAM
        part); the difference is host RAM. It does not say *which* GPU, so we
        attribute VRAM to the GPUs whose compute PIDs belong to the Ollama
        runner, splitting proportionally to each GPU's reported usage.
        """

        data = self._get("/api/ps")
        runner_pids = _ollama_runner_pids(gpus)
        models: list[LoadedModel] = []

        for entry in data.get("models", []):
            name = entry.get("name") or entry.get("model") or "?"
            total_mb = int(entry.get("size", 0)) // MIB
            vram_mb = int(entry.get("size_vram", 0)) // MIB
            ram_mb = max(0, total_mb - vram_mb)

            per_gpu = _attribute_vram(vram_mb, runner_pids, gpus)
            details = entry.get("details") or {}
            bits = [b for b in (details.get("parameter_size"), details.get("quantization_level")) if b]

            models.append(
                LoadedModel(
                    backend=self.name,
                    name=name,
                    handle=entry.get("model") or name,
                    # Ollama reports no layer count, so n_gpu_layers stays
                    # unknown; the RAM/VRAM byte split is the honest signal.
                    placement=PlacementConfig(gpu_ids=sorted(per_gpu)),
                    pid=next(iter(runner_pids), None),
                    vram_mb_per_gpu=per_gpu,
                    ram_mb=ram_mb,
                    size_mb=total_mb,
                    detail=" ".join(bits),
                )
            )
        return models

    def list_available(self) -> list[AvailableModel]:
        data = self._get("/api/tags")
        out: list[AvailableModel] = []
        for entry in data.get("models", []):
            name = entry.get("name") or entry.get("model")
            if not name:
                continue
            details = entry.get("details") or {}
            bits = [b for b in (details.get("parameter_size"), details.get("quantization_level")) if b]
            out.append(
                AvailableModel(
                    backend=self.name,
                    name=name,
                    handle=name,
                    size_mb=int(entry.get("size", 0)) // MIB,
                    detail=" ".join(bits),
                )
            )
        return sorted(out, key=lambda m: m.name)

    # --- control --------------------------------------------------------

    def load(self, model: AvailableModel, placement: PlacementConfig) -> None:
        """Warm a model into the requested split.

        An empty prompt makes Ollama load the model and return immediately
        instead of generating; the options ride along with the load.
        """

        options: dict[str, object] = {"main_gpu": placement.main_gpu}
        # num_gpu is layers-on-GPU; 0 keeps the model entirely in host RAM.
        options["num_gpu"] = 0 if placement.cpu_only else (placement.n_gpu_layers or 999)

        self._post(
            "/api/generate",
            {
                "model": model.handle,
                "prompt": "",
                "stream": False,
                "keep_alive": "30m",
                "options": options,
            },
        )

    def unload(self, model: LoadedModel) -> None:
        self._post(
            "/api/generate",
            {"model": model.handle, "prompt": "", "stream": False, "keep_alive": 0},
        )

    def supports_tensor_split(self) -> bool:
        return False


def _ollama_runner_pids(gpus: list[GPUInfo]) -> list[int]:
    """GPU compute PIDs that belong to an Ollama runner process."""
    pids: list[int] = []
    for gpu in gpus:
        for pid in gpu.pids:
            if pid in pids:
                continue
            try:
                name = " ".join(psutil.Process(pid).cmdline()).lower()
            except (psutil.Error, OSError):
                continue
            if "ollama" in name:
                pids.append(pid)
    return pids


def _attribute_vram(
    vram_mb: int, runner_pids: list[int], gpus: list[GPUInfo]
) -> dict[int, int]:
    """Spread a model's VRAM over the GPUs its runner actually occupies."""
    if vram_mb <= 0:
        return {}
    weights = {
        gpu.index: sum(mb for pid, mb in gpu.pids.items() if pid in runner_pids)
        for gpu in gpus
    }
    weights = {idx: mb for idx, mb in weights.items() if mb > 0}
    if not weights:
        return {}
    total = sum(weights.values())
    return {idx: round(vram_mb * mb / total) for idx, mb in weights.items()}
