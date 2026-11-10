"""The interactive Textual application."""

from __future__ import annotations

from rich.markup import escape

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
    Static,
)

from vramctl.backends.base import BackendError
from vramctl.config import (
    AvailableModel,
    ClusterState,
    GPUInfo,
    LoadedModel,
    NodeState,
    PlacementConfig,
)
from vramctl.manager import ClusterManager


def _gb(mb: int) -> str:
    return f"{mb / 1024:.1f}G" if mb else "0G"


SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def _sparkline(samples: list[int], width: int = 24) -> str:
    """Utilization history as block characters, oldest to newest.

    Scaled against 100% rather than the window's own maximum, so a quiet card
    stays visibly quiet instead of being rescaled into looking busy.
    """
    if not samples:
        return ""
    recent = samples[-width:]
    out = []
    for value in recent:
        level = min(len(SPARK) - 1, max(0, round(value / 100 * (len(SPARK) - 1))))
        out.append(SPARK[level])
    return "".join(out)


# Distinct, stable colours for models sharing a card. Chosen so neighbouring
# segments stay distinguishable in both light and dark terminals.
MODEL_COLORS = ("cyan", "magenta", "green", "yellow", "blue", "bright_cyan")
OTHER_COLOR = "bright_black"


def _model_color(name: str) -> str:
    """A colour that stays with a model across refreshes and restarts."""
    return MODEL_COLORS[sum(name.encode()) % len(MODEL_COLORS)]


def _meter(segments: list[tuple[int, str]], total: int, width: int) -> str:
    """A proportional bar: one coloured run per segment, then free space.

    Cells are placed from a running total rather than rounded individually,
    so the segments cannot drift past the bar's width.
    """
    if total <= 0 or width <= 0:
        return ""
    out: list[str] = []
    cumulative = 0
    filled = 0
    for mb, color in segments:
        if mb <= 0:
            continue
        cumulative += mb
        edge = min(width, round(width * cumulative / total))
        cells = edge - filled
        # Something resident should never render as nothing at all.
        if cells <= 0 and filled < width:
            cells = 1
            edge = filled + 1
        if cells > 0:
            out.append(f"[{color}]{'\u2588' * cells}[/]")
            filled = edge
    if filled < width:
        out.append(f"[dim]{'\u2591' * (width - filled)}[/]")
    return "".join(out)


def _slug(name: str) -> str:
    """A DOM-id-safe form of a machine name."""
    return "".join(c if c.isalnum() else "-" for c in name) or "node"


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
        /* A node with one GPU would otherwise stretch its card across the
           terminal, stranding a very long bar beside two short ones. */
        max-width: 72;
        height: auto;
        min-height: 7;
    }
    GpuCard:focus { border: round $accent; }
    GpuCard .gpu-title { text-style: bold; }
    GpuCard .gpu-meter { color: $text; }
    GpuCard .gpu-legend { color: $text-muted; }
    GpuCard .gpu-trend { color: $text-muted; }
    """

    # Room for the "VRAM "/"util " prefix and the trailing " 100%".
    GUTTER = 10
    MIN_BAR = 10
    # A fixed allowance for the sensor readout, so cards whose sensor strings
    # differ in length still line their meters up with each other.
    SENSOR_BUDGET = 26

    def __init__(
        self,
        gpu: GPUInfo,
        models: list[LoadedModel],
        series: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.gpu = gpu
        self._title = Label(classes="gpu-title")
        self._vram = Label(classes="gpu-meter")
        self._legend = Label(classes="gpu-legend")
        self._trend = Label(classes="gpu-trend")
        self._last: tuple = (gpu, models, series)
        self.set_state(gpu, models, series)

    def compose(self) -> ComposeResult:
        yield self._title
        yield self._vram
        yield self._legend
        yield self._trend

    def on_resize(self) -> None:
        """Re-draw at the new width; the bars are sized to the card."""
        self.set_state(*self._last)

    @property
    def _card_width(self) -> int:
        return self.content_size.width or 44

    def _meter_width(self, sensors: str = "") -> int:
        """One width for both meters, so their readings share a column.

        The sensors sit to the right of the utilization bar, so they set the
        width for the VRAM bar too rather than letting the two drift apart.
        """
        spare = self._card_width - self.GUTTER
        if sensors:
            spare -= max(self.SENSOR_BUDGET, len(sensors)) + 3
        return max(self.MIN_BAR, spare)

    def _title_text(self, gpu: GPUInfo, stats: list[str]) -> str:
        """Fit the figures beside the name, dropping detail as space runs out."""
        base = f"GPU {gpu.index}  {gpu.short_name}"
        for text in stats:
            if len(base) + 4 + len(text) <= self._card_width:
                return f"{base}    {text}"
        return base

    def set_state(
        self,
        gpu: GPUInfo,
        models: list[LoadedModel],
        series: list[int] | None = None,
    ) -> None:
        self.gpu = gpu
        self._last = (gpu, models, series)
        sensors = "" if gpu.declared else gpu.sensors
        width = self._meter_width(sensors)

        held = [
            (m, m.vram_mb_per_gpu.get(gpu.index, 0))
            for m in models
            if m.vram_mb_per_gpu.get(gpu.index, 0) > 0
        ]
        held.sort(key=lambda pair: -pair[1])

        if gpu.declared:
            # Nothing was measured, so the only real figure is what the
            # backends admit to holding.
            total = gpu.memory_total_mb
            by_models = sum(mb for _, mb in held)
            self._title.update(
                self._title_text(
                    gpu,
                    [
                        f"{_gb(by_models)} held / {_gb(total)} declared",
                        f"{_gb(by_models)}/{_gb(total)}",
                    ],
                )
            )
            segments = [(mb, _model_color(m.name)) for m, mb in held]
            pct = round(100 * by_models / total) if total else 0
            self._vram.update(f"VRAM {_meter(segments, total, width)} {pct:>3}%")
            self._trend.update("[dim]util unknown (host not measured)[/]")
        else:
            total = gpu.memory_total_mb
            used = gpu.memory_used_mb
            # Whatever holds VRAM but is not a model we can name: a compositor,
            # a game, someone else's process. Showing it beats a silent gap
            # between the model sizes and the card's real usage.
            other = max(0, used - sum(mb for _, mb in held))
            self._title.update(
                self._title_text(
                    gpu,
                    [
                        f"{_gb(used)} / {_gb(total)}  ({_gb(gpu.memory_free_mb)} free)",
                        f"{_gb(used)} / {_gb(total)}",
                        f"{_gb(gpu.memory_free_mb)} free",
                    ],
                )
            )
            segments = [(mb, _model_color(m.name)) for m, mb in held]
            if other > 0:
                segments.append((other, OTHER_COLOR))
            pct = round(gpu.used_fraction * 100)
            self._vram.update(f"VRAM {_meter(segments, total, width)} {pct:>3}%")

            samples = (series or [])[-width:]
            spark = _sparkline(samples, width)
            # Pad the unfilled part of the window so the reading does not
            # slide left as samples accumulate.
            pad = max(0, width - len(samples))
            graph = (f"[dim]{'\u00b7' * pad}[/]" if pad else "") + spark
            trend = f"util {graph} {gpu.utilization_pct:>3}%"
            if sensors:
                trend += f"   [dim]{sensors}[/]"
            self._trend.update(trend)

        self._legend.update(self._legend_text(held, gpu))

    def _legend_text(self, held: list, gpu: GPUInfo) -> str:
        """Name each coloured run, so the bar is readable without a key."""
        lines = []
        for model, mb in held[:3]:
            swatch = f"[{_model_color(model.name)}]\u25a0[/]"
            lines.append(f"  {swatch} {escape(model.name)}  {_gb(mb)}")
        if len(held) > 3:
            rest = sum(mb for _, mb in held[3:])
            lines.append(f"  [dim]+{len(held) - 3} more  {_gb(rest)}[/]")
        if not gpu.declared:
            other = max(0, gpu.memory_used_mb - sum(mb for _, mb in held))
            if other > 0:
                # A card can be busy with things that are not models at all;
                # calling that "idle" is what made the bar look wrong.
                lines.append(
                    f"  [{OTHER_COLOR}]\u25a0[/] [dim]other processes  "
                    f"{_gb(other)}[/]"
                )
        if not lines:
            return "  [dim](idle)[/]"
        return "\n".join(lines)


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

    SPREAD_MARK = {"vram": "\u25b0\u25b0", "split": "\u25b0\u25b1", "ram": "\u25b1\u25b1"}

    def set_model(self, model: LoadedModel) -> None:
        self.model = model
        where = model.location
        pid = f" \u00b7 pid {model.pid}" if model.pid else ""
        detail = f" \u00b7 {model.detail}" if model.detail else ""
        endpoint = f" \u00b7 {model.endpoint}" if model.endpoint else ""
        # The marker describes how a *model* is placed. Another process always
        # has RSS, so showing it as "split" would say nothing.
        mark = self.SPREAD_MARK.get(model.spread, "") if model.manageable else "\u2022"
        self._main.update(
            f"{mark} {model.node}  [{model.backend}] {model.name}"
            if model.node
            else f"{mark} [{model.backend}] {model.name}"
        )
        split = f"VRAM {_gb(model.vram_mb)} \u00b7 RAM {_gb(model.ram_mb)}"
        # Share of what is actually resident, not of the file on disk: VRAM
        # also holds the KV cache, compute buffers and CUDA context, so
        # measuring against the file size can exceed 100%.
        resident = model.vram_mb + model.ram_mb
        if model.spread == "split" and resident:
            pct = round(100 * model.vram_mb / resident)
            split += f" ({pct}% on GPU)"
        self._sub.update(f"    {where} \u00b7 {split}{endpoint}{pid}{detail}")


class GpuToggleItem(ListItem):
    """A GPU in the modal's target list, checkable with space."""

    def __init__(self, gpu: GPUInfo, selected: bool, show_node: bool = False) -> None:
        super().__init__()
        self.gpu = gpu
        self.selected = selected
        self.show_node = show_node
        self._label = Label()
        self._refresh()

    def compose(self) -> ComposeResult:
        yield self._label

    def toggle(self) -> None:
        self.selected = not self.selected
        self._refresh()

    def _refresh(self) -> None:
        mark = "x" if self.selected else " "
        where = f"{self.gpu.node} " if self.show_node and self.gpu.node else ""
        free = (
            f"{_gb(self.gpu.memory_total_mb)} declared"
            if self.gpu.declared
            else f"{_gb(self.gpu.memory_free_mb)} free"
        )
        self._label.update(
            f"[{mark}] {where}GPU {self.gpu.index}  {self.gpu.short_name}  ({free})"
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
    .modal-target { color: $accent; text-style: bold; }
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
        self._multi_node = len({g.node for g in gpus if g.node}) > 1

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(self.title_text, id="modal-title")

            if self.fixed_model is None:
                yield Label("Model (\u2191/\u2193 to choose):", classes="modal-hint")
                yield ListView(
                    *[
                        ListItem(
                            Label(
                                f"[{m.backend}] {m.name}  {_gb(m.size_mb)}"
                                + (f"   {m.node}" if m.node else "")
                                + (f" \u00b7 {m.endpoint}" if m.endpoint else "")
                            )
                        )
                        for m in self.choices
                    ],
                    id="modal-models",
                )
            else:
                yield Label(
                    f"Model: [{self.fixed_model.backend}] {self.fixed_model.name}",
                    classes="modal-hint",
                )

            yield Label(id="modal-target", classes="modal-target")
            yield Label("Target GPUs (space to toggle):", classes="modal-hint")
            yield ListView(
                *[
                    GpuToggleItem(
                        gpu,
                        gpu.index in self.initial.gpu_ids,
                        show_node=self._multi_node,
                    )
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
        self._render_target()
        if self.fixed_model is None and self.choices:
            self.query_one("#modal-models", ListView).focus()
        elif self.gpus:
            self.query_one("#modal-gpus", ListView).focus()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "modal-models":
            self._render_target()

    def _render_target(self) -> None:
        """Say where the load will actually land.

        For a pinned Ollama endpoint the card is not up for selection - the
        endpoint already decides it - so the GPU toggles do not apply.
        """
        try:
            label = self.query_one("#modal-target", Label)
        except Exception:
            return
        model = self._chosen_model()
        if model is None:
            label.update("Target: pick a model")
            return
        if self.fixed_model is not None and model.backend == "ollama":
            picked = self.selected_gpus
            if picked:
                label.update(
                    f"Target: {model.node} \u2192 GPU {picked[0]} "
                    f"(switches to that card's endpoint; unload + reload)"
                )
            else:
                label.update(
                    f"Target: {model.node} \u2192 RAM only (unload + reload)"
                )
            return
        if model.backend == "ollama":
            where = model.node or "local"
            if model.pinned_gpu is not None:
                label.update(
                    f"Target: {where} \u00b7 {model.endpoint} \u2192 "
                    f"GPU {model.pinned_gpu} (fixed by the endpoint; "
                    f"toggles below do not apply)"
                )
            else:
                label.update(
                    f"Target: {where} \u00b7 {model.endpoint} "
                    f"(unpinned; main_gpu follows your selection)"
                )
            return
        label.update(f"Target: {model.node or 'local'} \u00b7 {model.backend}")

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
            self._render_target()
            event.stop()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Enter on the GPU list toggles; enter on the model list confirms."""
        if event.list_view.id == "modal-gpus":
            if isinstance(event.item, GpuToggleItem):
                event.item.toggle()
                self._render_target()
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
        if model.backend == "ollama" and model.pinned_gpu is not None:
            # The endpoint is the card; recording anything else would make the
            # panel disagree with where the model lands.
            gpu_ids = [model.pinned_gpu]

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
                node=self.fixed_model.node,
                endpoint=self.fixed_model.endpoint,
            )
        if not self.choices:
            return None
        view = self.query_one("#modal-models", ListView)
        index = view.index
        if index is None or index >= len(self.choices):
            return None
        return self.choices[index]


class ConfirmStopModal(ModalScreen[bool]):
    """Confirm stopping a process before any signal is sent.

    Stopping is irreversible and may be on another machine, so it is never
    the default: the dialog opens on Cancel and needs an explicit choice.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("n", "cancel", "No"),
        Binding("y", "confirm", "Yes"),
    ]

    DEFAULT_CSS = """
    ConfirmStopModal { align: center middle; }
    #confirm-box {
        width: 72;
        height: auto;
        border: round $error;
        background: $surface;
        padding: 1 2;
    }
    #confirm-title { text-style: bold; color: $error; margin-bottom: 1; }
    .confirm-line { color: $text; }
    .confirm-hint { color: $text-muted; margin-top: 1; }
    """

    def __init__(self, model: LoadedModel, remote: bool) -> None:
        super().__init__()
        self.model = model
        self.remote = remote

    def compose(self) -> ComposeResult:
        m = self.model
        with Vertical(id="confirm-box"):
            yield Label("Stop this process?", id="confirm-title")
            yield Label(f"  {m.name}  (pid {m.pid})", classes="confirm-line")
            yield Label(
                f"  on {m.node}" + ("  \u2014 over SSH" if self.remote else "  \u2014 locally"),
                classes="confirm-line",
            )
            yield Label(
                f"  holding {_gb(m.vram_mb)} VRAM on {m.location}",
                classes="confirm-line",
            )
            if not m.manageable:
                yield Label(
                    "  This is not a model vramctl manages. Whatever it is "
                    "doing will end.",
                    classes="confirm-line",
                )
            yield Label(
                "SIGTERM, then SIGKILL if it ignores that.  y: stop  \u00b7  "
                "n / esc: cancel",
                classes="confirm-hint",
            )

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)


class VramctlApp(App[None]):
    """Arrow-key dashboard over GPUs, RAM, and loaded models."""

    TITLE = "vramctl"

    CSS = """
    #nodes { height: auto; }
    .gpu-row { height: auto; }
    .node-header { padding: 0 1; text-style: bold; color: $accent; }
    .node-empty { padding: 0 1; color: $text-muted; }
    #models-title { padding: 0 1; text-style: bold; }
    #models { height: 1fr; border: round $panel-lighten-2; margin: 0 1; }
    #models:focus-within { border: round $accent; }
    #status { padding: 0 1; color: $text-muted; }
    """

    BINDINGS = [
        Binding("l", "load", "Load"),
        Binding("s", "split", "Split/Move"),
        Binding("u", "unload", "Unload"),
        Binding("k", "stop_process", "Stop proc"),
        Binding("r", "refresh", "Refresh"),
        Binding("left", "focus_prev_gpu", "Prev GPU", show=False),
        Binding("right", "focus_next_gpu", "Next GPU", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, manager: ClusterManager | None = None) -> None:
        super().__init__()
        self.manager = manager or ClusterManager()
        self._busy = False
        # Cards are built once per (node, gpu) and updated in place; remounting
        # them every tick would drop keyboard focus mid-poll.
        self._cards: dict[tuple[str, int], GpuCard] = {}
        self._node_shape: list = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(
            Vertical(id="nodes"),
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
    def refresh_state(self, force: bool = False) -> None:
        """Poll off the UI thread; nvidia-smi, ssh and HTTP all block."""
        state = self.manager.refresh_state(force_telemetry=force)
        self.call_from_thread(self._render, state)

    def _render(self, state: ClusterState) -> None:
        self._render_nodes(state)
        self._render_models(state)

        status = self.query_one("#status", Label)
        if state.errors:
            status.update("\u26a0 " + "  ".join(state.errors[:2]))
        else:
            count = len(state.managed)
            others = len(state.models) - count
            gpus = len(state.gpus)
            line = (
                f"{count} model{'' if count == 1 else 's'} across "
                f"{gpus} GPU{'' if gpus == 1 else 's'} on "
                f"{len(state.nodes)} machine{'' if len(state.nodes) == 1 else 's'}"
            )
            if others:
                line += f"  \u00b7  {others} other VRAM consumer{'' if others == 1 else 's'}"
            status.update(line)

    def _node_header(self, node: NodeState) -> str:
        bits = [node.name]
        if node.ram:
            bits.append(
                f"RAM {_gb(node.ram.used_mb)}/{_gb(node.ram.total_mb)}"
            )
        if not node.reachable:
            bits.append("\u26a0 unreachable, showing declared capacity")
        return "  \u00b7  ".join(bits)

    def _render_nodes(self, state: ClusterState) -> None:
        """Render one section per machine, remounting only if the shape changed."""
        container = self.query_one("#nodes", Vertical)
        shape = [(n.name, tuple(g.index for g in n.gpus)) for n in state.nodes]

        if shape == self._node_shape and shape:
            for node in state.nodes:
                self.query_one(f"#hdr-{_slug(node.name)}", Label).update(
                    self._node_header(node)
                )
                for gpu in node.gpus:
                    card = self._cards.get((node.name, gpu.index))
                    if card is not None:
                        card.set_state(
                            gpu,
                            state.models_on_gpu(node.name, gpu.index),
                            self.manager.series_for(node.name, gpu.index),
                        )
            return

        container.remove_children()
        self._cards.clear()
        if not state.nodes:
            container.mount(Label("  No machines configured"))
            self._node_shape = shape
            return

        for node in state.nodes:
            container.mount(
                Label(self._node_header(node), id=f"hdr-{_slug(node.name)}",
                      classes="node-header")
            )
            if not node.gpus:
                container.mount(Label("    (no GPUs reported)", classes="node-empty"))
                continue
            row = Horizontal(classes="gpu-row")
            container.mount(row)
            cards = []
            for gpu in node.gpus:
                card = GpuCard(
                    gpu,
                    state.models_on_gpu(node.name, gpu.index),
                    self.manager.series_for(node.name, gpu.index),
                )
                self._cards[(node.name, gpu.index)] = card
                cards.append(card)
            row.mount_all(cards)
        self._node_shape = shape

    def _render_models(self, state: ClusterState) -> None:
        """Update the model list, preserving the cursor across refreshes."""
        view = self.query_one("#models", ListView)
        rows = [r for r in view.children if isinstance(r, ModelRow)]
        keys = [m.key for m in state.models]

        if [r.model.key for r in rows] == keys and rows:
            for row, model in zip(rows, state.models):
                row.set_model(model)
            return

        selected_key = (
            rows[view.index].model.key
            if rows and view.index is not None and view.index < len(rows)
            else None
        )
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
        """Manual refresh re-reads remote hardware instead of using the cache."""
        self.refresh_state(force=True)

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
        if not model.manageable:
            self.notify(
                f"{model.name} is not a model vramctl manages - it is another "
                f"process holding VRAM on {model.node}",
                severity="warning",
            )
            return
        # Moving a model is only meaningful within its own machine.
        same_node = [g for g in self.manager.state.gpus if g.node == model.node]
        self.push_screen(
            LoadOrSplitModal(
                f"Move or re-split {model.name} on {model.node}",
                same_node or self.manager.state.gpus,
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
        if not model.manageable:
            self.notify(
                f"{model.name} is not a model vramctl manages - press 'k' to "
                f"stop the process on {model.node} instead",
                severity="warning",
            )
            return
        self._run_unload(model)

    def action_stop_process(self) -> None:
        """Stop the process behind the selected row, after confirmation.

        Deliberately a separate key from unload: `u` asks a backend to release
        a model, while this kills a process, possibly on another machine.
        """
        model = self.selected_model
        if model is None:
            self.notify("Select a row first", severity="warning")
            return
        if not self.manager.config.allow_process_kill:
            self.notify(
                "Stopping processes is disabled (allow_process_kill is false)",
                severity="warning",
            )
            return
        if model.pid is None:
            self.notify(
                f"{model.name} has no pid - it is reported by an API, not a "
                f"process. Use 'u' to unload it.",
                severity="warning",
            )
            return

        node = next(
            (n for n in self.manager.nodes if n.name == model.node), None
        )
        if node is None or not node.measurable:
            self.notify(f"No way to reach {model.node}", severity="error")
            return

        self.push_screen(
            ConfirmStopModal(model, remote=bool(node.ssh) and not node.local),
            lambda ok: self._run_stop(model) if ok else None,
        )

    @work(exclusive=True, thread=True)
    def _run_stop(self, model: LoadedModel) -> None:
        self._busy = True
        self.call_from_thread(
            self._set_status, f"Stopping {model.name} (pid {model.pid}) on {model.node}\u2026"
        )
        try:
            result = self.manager.stop_process(model)
            self.call_from_thread(self.notify, result)
        except BackendError as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
        finally:
            self._busy = False
            self.call_from_thread(self._render, self.manager.state)

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
