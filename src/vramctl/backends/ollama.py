"""Ollama adapter: one instance per endpoint.

An Ollama instance may be pinned to a single GPU by its systemd unit - on
InferencePrime each one runs under ``DevicePolicy=closed`` with a single
``DeviceAllow``, so it can open exactly one card. Two consequences shape this
adapter:

* The endpoint *is* the GPU. Everything the instance reports lives on that
  card, so VRAM needs no PID correlation to attribute.
* Inside such an instance the one visible card enumerates as CUDA device 0.
  A load must therefore address it as ``main_gpu: 0``; sending the physical
  index would name a device that does not exist in that cgroup.
"""

from __future__ import annotations

import httpx

from vramctl.backends.base import BackendError
from vramctl.config import (
    MIB,
    OLLAMA,
    AvailableModel,
    Config,
    EndpointConfig,
    GPUInfo,
    LoadedModel,
    PlacementConfig,
)


class OllamaBackend:
    name = OLLAMA

    def __init__(
        self,
        config: Config,
        endpoint: EndpointConfig | None = None,
        node: str = "",
    ) -> None:
        self.config = config
        self.endpoint = endpoint or EndpointConfig(url=config.ollama_url)
        self.base_url = self.endpoint.url.rstrip("/")
        self.node = node
        self.timeout = config.request_timeout

    @property
    def label(self) -> str:
        return self.endpoint.name

    def is_enabled(self) -> bool:
        return self.config.ollama_enabled

    @property
    def pinned_gpu(self) -> int | None:
        return self.endpoint.gpu

    # --- transport ------------------------------------------------------

    def _get(self, path: str) -> dict:
        try:
            resp = httpx.get(f"{self.base_url}{path}", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama {self.label}{path}: {exc}") from exc
        except ValueError as exc:
            raise BackendError(f"ollama {self.label}{path}: invalid JSON") from exc

    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama {self.label}{path}: {exc}") from exc
        except ValueError:
            return {}

    # --- inspection -----------------------------------------------------

    def list_running(self, gpus: list[GPUInfo]) -> list[LoadedModel]:
        """Map ``GET /api/ps`` onto ``LoadedModel``.

        Ollama reports ``size`` (total resident) and ``size_vram`` (the VRAM
        part); the difference is host RAM, which is how a partially offloaded
        model shows up as split rather than resident.
        """

        data = self._get("/api/ps")
        models: list[LoadedModel] = []

        for entry in data.get("models", []):
            name = entry.get("name") or entry.get("model") or "?"
            total_mb = int(entry.get("size", 0)) // MIB
            vram_mb = int(entry.get("size_vram", 0)) // MIB
            ram_mb = max(0, total_mb - vram_mb)

            per_gpu: dict[int, int] = {}
            if vram_mb > 0:
                # The endpoint identifies the card, so no attribution guesswork.
                target = self.pinned_gpu if self.pinned_gpu is not None else 0
                per_gpu[target] = vram_mb

            details = entry.get("details") or {}
            bits = [
                b
                for b in (
                    details.get("parameter_size"),
                    details.get("quantization_level"),
                )
                if b
            ]
            ctx = entry.get("context_length")
            if isinstance(ctx, int) and ctx > 0:
                bits.append(f"{ctx // 1024}k ctx" if ctx >= 1024 else f"{ctx} ctx")

            models.append(
                LoadedModel(
                    backend=self.name,
                    name=name,
                    handle=entry.get("model") or name,
                    # Ollama reports no layer count; the byte split is the
                    # honest signal for how the model is placed.
                    placement=PlacementConfig(gpu_ids=sorted(per_gpu)),
                    vram_mb_per_gpu=per_gpu,
                    ram_mb=ram_mb,
                    size_mb=total_mb,
                    detail=" ".join(bits),
                    node=self.node,
                    endpoint=self.label,
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
            bits = [
                b
                for b in (
                    details.get("parameter_size"),
                    details.get("quantization_level"),
                )
                if b
            ]
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

        An empty prompt makes Ollama load the model and return rather than
        generate, so the options ride along with the load.
        """

        options: dict[str, object] = {
            # 0 keeps the model entirely in host RAM; otherwise offload all
            # layers unless a specific count was asked for.
            "num_gpu": 0 if placement.cpu_only else (placement.n_gpu_layers or 999),
            "main_gpu": self._main_gpu(placement),
        }
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

    def _main_gpu(self, placement: PlacementConfig) -> int:
        """Which device index this instance should use.

        A pinned instance sees exactly one card, which is device 0 inside its
        cgroup regardless of the card's physical index. Only an unpinned
        instance, which sees every GPU, can take a physical index.
        """
        if self.pinned_gpu is not None:
            return 0
        return placement.main_gpu

    def unload(self, model: LoadedModel) -> None:
        self._post(
            "/api/generate",
            {"model": model.handle, "prompt": "", "stream": False, "keep_alive": 0},
        )

    def supports_tensor_split(self) -> bool:
        """Ollama exposes num_gpu and main_gpu only - no per-device ratio."""
        return False
