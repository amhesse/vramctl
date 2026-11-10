"""The interactive Textual application."""

from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Static,
)

from vramctl.backends.base import BackendError
from vramctl.config import (
    AvailableModel,
    ClusterState,
    GPUInfo,
    LoadedModel,
    PlacementConfig,
)
from vramctl.manager import ClusterManager


def _gb(mb: int) -> str:
    return f"{mb / 1024:.1f}G" if mb else "0G"


class GpuCard(Static):
    """One GPU: name, VRAM meter, and the models resident on it.

    Built once per GPU and updated in place — remounting it on every refresh
    would drop keyboard focus mid-poll.
    """

    can_focus = True

    DEFAULT_CSS = """
    GpuCard {
        border: round $panel-lighten-2;
        padding: 0 1;
        margin: 0 1 1 0;
        width: 1fr;
        height: auto;
        min-height: 7;
    }
    GpuCard:focus { border: round $accent; }
    GpuCard .gpu-title { text-style: bold; }
    GpuCard .gpu-mem { color: $text-muted; }
    GpuCard .vramctl { color: $text-muted; }
    """

    def __init__(self, gpu: GPUInfo, models: list[LoadedModel]) -> None:
        super().__init__()
        self.gpu = gpu
        self._title = Label(classes="gpu-title")
        self._mem = Label(classes="gpu-mem")
        self._bar = ProgressBar(total=100, show_eta=False, show_percentage=True)
        self._models = Label(classes="vramctl")
        self.set_state(gpu, models)

    def compose(self) -> ComposeResult:
        yield self._title
        yield self._mem
        yield self._bar
        yield self._models

    def set_state(self, gpu: GPUInfo, models: list[LoadedModel]) -> None:
        self.gpu = gpu
        self._title.update(f"GPU {gpu.index}  {gpu.short_name}")
        self._mem.update(
            f"{_gb(gpu.memory_used_mb)} / {_gb(gpu.memory_total_mb)} VRAM"
            f"   {gpu.utilization_pct}% util"
        )
        self._bar.update(progress=round(gpu.used_fraction * 100))
        if models:
            lines = "\n".join(
                f"  \u2022 {m.name}  {_gb(m.vram_mb_per_gpu.get(gpu.index, 0))}"
                for m in models
            )
        else:
            lines = "  (idle)"
        self._models.update(lines)


class ModelRow(ListItem):
    """One loaded model, with its placement and memory split."""

    DEFAULT_CSS = """
    ModelRow { padding: 0 1; height: auto; }
    ModelRow .row-main { text-style: bold; }
    ModelRow .row-sub { color: $text-muted; }
    """

    def __init__(self, model: LoadedModel) -> None:
        super().__init__()
        self._main = Label(classes="row-main")
        self._sub = Label(classes="row-sub")
        self.set_model(model)

    def compose(self) -> ComposeResult:
        yield self._main
        yield self._sub

    def set_model(self, model: LoadedModel) -> None:
        self.model = model
        where = (
            "+".join(f"GPU{i}" for i in model.gpu_ids) if model.gpu_ids else "RAM only"
        )
        pid = f" \u00b7 pid {model.pid}" if model.pid else ""
        detail = f" \u00b7 {model.detail}" if model.detail else ""
        self._main.update(f"[{model.backend}] {model.name}")
        self._sub.update(
            f"    {where} \u00b7 VRAM {_gb(model.vram_mb)} \u00b7 "
            f"RAM {_gb(model.ram_mb)}{pid}{detail}"
        )


class GpuToggleItem(ListItem):
    """A GPU in the modal's target list, checkable with space."""

    def __init__(self, gpu: GPUInfo, selected: bool) -> None:
        super().__init__()
        self.gpu = gpu
        self.selected = selected
        self._label = Label()
        self._refresh()

    def compose(self) -> ComposeResult:
        yield self._label

    def toggle(self) -> None:
        self.selected = not self.selected
        self._refresh()

    def _refresh(self) -> None:
        mark = "x" if self.selected else " "
        self._label.update(
            f"[{mark}] GPU {self.gpu.index}  {self.gpu.short_name}  "
            f"({_gb(self.gpu.memory_free_mb)} free)"
        )


class LoadOrSplitModal(ModalScreen[tuple[AvailableModel, PlacementConfig] | None]):
    """Pick a model (load) or keep the current one (split), then place it.

    Returns the chosen model plus its placement, or ``None`` if dismissed.
    Enter confirms from anywhere; space checks the GPU under the cursor.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("tab", "focus_next", "Next field", show=False),
    ]

    DEFAULT_CSS = """
    LoadOrSplitModal { align: center middle; }
    #modal-box {
        width: 80%;
        max-width: 90;
        height: auto;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    #modal-title { text-style: bold; margin-bottom: 1; }
    #modal-models { height: 8; border: round $panel-lighten-2; }
    #modal-gpus { height: auto; max-height: 6; border: round $panel-lighten-2; }
    .modal-hint { color: $text-muted; }
    """

    def __init__(
        self,
        title: str,
        gpus: list[GPUInfo],
        choices: list[AvailableModel] | None,
        fixed_model: LoadedModel | None = None,
        initial: PlacementConfig | None = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.gpus = gpus
        self.choices = choices or []
        self.fixed_model = fixed_model
        self.initial = initial or PlacementConfig(
            gpu_ids=[g.index for g in gpus[:1]]
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(self.title_text, id="modal-title")

            if self.fixed_model is None:
                yield Label("Model (\u2191/\u2193 to choose):", classes="modal-hint")
                yield ListView(
                    *[
                        ListItem(Label(f"[{m.backend}] {m.name}  {_gb(m.size_mb)}"))
                        for m in self.choices
                    ],
                    id="modal-models",
                )
            else:
                yield Label(
                    f"Model: [{self.fixed_model.backend}] {self.fixed_model.name}",
                    classes="modal-hint",
                )

            yield Label("Target GPUs (space to toggle):", classes="modal-hint")
            yield ListView(
                *[
                    GpuToggleItem(gpu, gpu.index in self.initial.gpu_ids)
                    for gpu in self.gpus
                ],
                id="modal-gpus",
            )

            yield Label("GPU layers (blank = all, 0 = RAM only):", classes="modal-hint")
            yield Input(
                value=(
                    ""
                    if self.initial.n_gpu_layers is None
                    else str(self.initial.n_gpu_layers)
                ),
                placeholder="all",
                id="modal-layers",
            )
            yield Label(
                "Tensor split, e.g. 24,16 (blank = even):", classes="modal-hint"
            )
            yield Input(
                value=(
                    ",".join(f"{w:g}" for w in self.initial.tensor_split)
                    if self.initial.tensor_split
                    else ""
                ),
                placeholder="even",
                id="modal-split",
            )
            yield Label(
                "enter: confirm \u00b7 space: toggle GPU \u00b7 tab: next field \u00b7 esc: cancel",
                classes="modal-hint",
            )

    def on_mount(self) -> None:
        if self.fixed_model is None and self.choices:
            self.query_one("#modal-models", ListView).focus()
        elif self.gpus:
            self.query_one("#modal-gpus", ListView).focus()

    @property
    def selected_gpus(self) -> list[int]:
        return sorted(
            item.gpu.index
            for item in self.query(GpuToggleItem)
            if item.selected
        )

    def on_key(self, event) -> None:
        if event.key != "space":
            return
        gpu_list = self.query_one("#modal-gpus", ListView)
        if self.focused is not gpu_list:
            return
        item = gpu_list.highlighted_child
        if isinstance(item, GpuToggleItem):
            item.toggle()
            event.stop()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Enter on the GPU list toggles; enter on the model list confirms."""
        if event.list_view.id == "modal-gpus":
            if isinstance(event.item, GpuToggleItem):
                event.item.toggle()
            return
        self.action_confirm()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_confirm()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        model = self._chosen_model()
        if model is None:
            self.notify("Select a model first", severity="warning")
            return

        gpu_ids = self.selected_gpus

        layers_raw = self.query_one("#modal-layers", Input).value.strip()
        layers: int | None = None
        if layers_raw:
            try:
                layers = int(layers_raw)
            except ValueError:
                self.notify("GPU layers must be a whole number", severity="error")
                return
            if layers < 0:
                self.notify("GPU layers cannot be negative", severity="error")
                return

        split_raw = self.query_one("#modal-split", Input).value.strip()
        split: list[float] | None = None
        if split_raw:
            try:
                split = [float(p) for p in split_raw.split(",") if p.strip()]
            except ValueError:
                self.notify(
                    "Tensor split must be comma-separated numbers", severity="error"
                )
                return
            if len(split) != len(gpu_ids):
                self.notify(
                    f"Tensor split needs {len(gpu_ids)} value(s) to match the "
                    f"selected GPUs",
                    severity="error",
                )
                return
            if any(w < 0 for w in split) or sum(split) <= 0:
                self.notify("Tensor split weights must be positive", severity="error")
                return

        if not gpu_ids and layers not in (None, 0):
            self.notify(
                "No GPU selected, so the layer count is ignored \u2014 "
                "clear it or pick a GPU",
                severity="warning",
            )

        placement = PlacementConfig(
            gpu_ids=gpu_ids,
            n_gpu_layers=layers,
            tensor_split=split,
            port=self.initial.port,
        )
        self.dismiss((model, placement))

    def _chosen_model(self) -> AvailableModel | None:
        if self.fixed_model is not None:
            return AvailableModel(
                backend=self.fixed_model.backend,
                name=self.fixed_model.name,
                handle=self.fixed_model.handle,
                size_mb=self.fixed_model.size_mb,
            )
        if not self.choices:
            return None
        view = self.query_one("#modal-models", ListView)
        index = view.index
        if index is None or index >= len(self.choices):
            return None
        return self.choices[index]


class VramctlApp(App[None]):
    """Arrow-key dashboard over GPUs, RAM, and loaded models."""

    TITLE = "vramctl"

    CSS = """
    #gpu-row { height: auto; }
    #ram-line { padding: 0 1; color: $text-muted; }
    #models-title { padding: 0 1; text-style: bold; }
    #models { height: 1fr; border: round $panel-lighten-2; margin: 0 1; }
    #models:focus-within { border: round $accent; }
    #status { padding: 0 1; color: $text-muted; }
    """

    BINDINGS = [
        Binding("l", "load", "Load"),
        Binding("s", "split", "Split/Move"),
        Binding("u", "unload", "Unload"),
        Binding("r", "refresh", "Refresh"),
        Binding("left", "focus_prev_gpu", "Prev GPU", show=False),
        Binding("right", "focus_next_gpu", "Next GPU", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, manager: ClusterManager | None = None) -> None:
        super().__init__()
        self.manager = manager or ClusterManager()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(
            Horizontal(id="gpu-row"),
            Label(id="ram-line"),
            Label("Loaded models", id="models-title"),
            ListView(id="models"),
            Label(id="status"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(self.manager.config.refresh_seconds, self._tick)
        self.action_refresh()

    # --- rendering ------------------------------------------------------

    def _tick(self) -> None:
        if not self._busy:
            self.refresh_state()

    @work(exclusive=True, thread=True)
    def refresh_state(self) -> None:
        """Poll off the UI thread; nvidia-smi and HTTP both block."""
        state = self.manager.refresh_state()
        self.call_from_thread(self._render, state)

    def _render(self, state: ClusterState) -> None:
        self._render_gpus(state)

        if state.ram:
            ram = state.ram
            self.query_one("#ram-line", Label).update(
                f"System RAM  {_gb(ram.used_mb)} / {_gb(ram.total_mb)} "
                f"({_gb(ram.available_mb)} available)"
            )

        self._render_models(state)

        status = self.query_one("#status", Label)
        if state.errors:
            status.update("\u26a0 " + "  ".join(state.errors))
        else:
            count = len(state.models)
            status.update(f"{count} model{'' if count == 1 else 's'} resident")

    def _render_gpus(self, state: ClusterState) -> None:
        """Update the GPU cards, remounting only if the GPU set changed."""
        row = self.query_one("#gpu-row", Horizontal)
        cards = list(self.query(GpuCard))
        indices = [g.index for g in state.gpus]

        if [c.gpu.index for c in cards] == indices and cards:
            for card in cards:
                card.set_state(
                    next(g for g in state.gpus if g.index == card.gpu.index),
                    state.models_on_gpu(card.gpu.index),
                )
            return

        row.remove_children()
        if state.gpus:
            row.mount_all(
                GpuCard(gpu, state.models_on_gpu(gpu.index)) for gpu in state.gpus
            )
        else:
            row.mount(Label("  No NVIDIA GPUs detected"))

    def _render_models(self, state: ClusterState) -> None:
        """Update the model list, preserving the cursor across refreshes."""
        view = self.query_one("#models", ListView)
        rows = [r for r in view.children if isinstance(r, ModelRow)]
        keys = [m.key for m in state.models]

        if [r.model.key for r in rows] == keys and rows:
            for row, model in zip(rows, state.models):
                row.set_model(model)
            return

        # The set changed, so rebuild — but keep the cursor on the same model
        # if it is still loaded.
        selected_key = rows[view.index].model.key if rows and view.index is not None and view.index < len(rows) else None
        view.clear()
        for model in state.models:
            view.append(ModelRow(model))
        if state.models:
            view.index = keys.index(selected_key) if selected_key in keys else 0

    @property
    def selected_model(self) -> LoadedModel | None:
        models = self.query_one("#models", ListView)
        item = models.highlighted_child
        return item.model if isinstance(item, ModelRow) else None

    # --- actions --------------------------------------------------------

    def action_refresh(self) -> None:
        self.refresh_state()

    def action_focus_next_gpu(self) -> None:
        self._cycle_gpu(1)

    def action_focus_prev_gpu(self) -> None:
        self._cycle_gpu(-1)

    def _cycle_gpu(self, step: int) -> None:
        cards = list(self.query(GpuCard))
        if not cards:
            return
        focused = self.focused
        current = cards.index(focused) if focused in cards else -1
        cards[(current + step) % len(cards)].focus()

    def action_load(self) -> None:
        state = self.manager.state
        choices = self.manager.list_available()
        if not choices:
            self.notify(
                "No loadable models found. Check Ollama and the configured "
                "model directories.",
                severity="warning",
            )
            return
        self.push_screen(
            LoadOrSplitModal("Load a model", state.gpus, choices),
            self._on_load_result,
        )

    def action_split(self) -> None:
        model = self.selected_model
        if model is None:
            self.notify("Select a loaded model first", severity="warning")
            return
        self.push_screen(
            LoadOrSplitModal(
                f"Move or re-split {model.name}",
                self.manager.state.gpus,
                choices=None,
                fixed_model=model,
                initial=model.placement,
            ),
            self._on_split_result,
        )

    def action_unload(self) -> None:
        model = self.selected_model
        if model is None:
            self.notify("Select a loaded model first", severity="warning")
            return
        self._run_unload(model)

    def _on_load_result(self, result) -> None:
        if result is None:
            return
        model, placement = result
        self._run_load(model, placement)

    def _on_split_result(self, result) -> None:
        if result is None:
            return
        _, placement = result
        model = self.selected_model
        if model is not None:
            self._run_resplit(model, placement)

    # --- background work ------------------------------------------------

    @work(exclusive=True, thread=True)
    def _run_load(self, model: AvailableModel, placement: PlacementConfig) -> None:
        self._busy = True
        self.call_from_thread(
            self._set_status, f"Loading {model.name} onto {placement.describe()}…"
        )
        try:
            self.manager.load_model(model, placement)
            self.call_from_thread(self.notify, f"Loaded {model.name}")
        except BackendError as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self._busy = False
            self.call_from_thread(self._render, self.manager.state)

    @work(exclusive=True, thread=True)
    def _run_unload(self, model: LoadedModel) -> None:
        self._busy = True
        self.call_from_thread(self._set_status, f"Unloading {model.name}…")
        try:
            self.manager.unload_model(model)
            self.call_from_thread(self.notify, f"Unloaded {model.name}")
        except BackendError as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self._busy = False
            self.call_from_thread(self._render, self.manager.state)

    @work(exclusive=True, thread=True)
    def _run_resplit(self, model: LoadedModel, placement: PlacementConfig) -> None:
        self._busy = True
        self.call_from_thread(
            self._set_status,
            f"Re-splitting {model.name} → {placement.describe()} (unload + reload)…",
        )
        try:
            self.manager.resplit_model(model, placement)
            self.call_from_thread(
                self.notify, f"{model.name} now on {placement.describe()}"
            )
        except BackendError as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self._busy = False
            self.call_from_thread(self._render, self.manager.state)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Label).update(text)
