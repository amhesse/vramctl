# vramctl

An interactive terminal UI (TUI) for inspecting, loading, unloading, and splitting LLM workloads across NVIDIA GPUs and system RAM. Supports **Ollama**, **llama.cpp**, and **Strata**.

## Approach & Architecture

`vramctl` combines hardware-level process inspection (`nvidia-smi` and `/proc/<pid>`) with runtime control APIs and subprocess management:

1. **Hardware & Process Correlation (`src/vramctl/nvidia.py`)**
   Polls `nvidia-smi` for per-GPU VRAM usage and active compute PIDs, alongside system RAM metrics via `psutil`.
2. **Backend Adapters (`src/vramctl/backends/`)**
   - **Ollama (`src/vramctl/backends/ollama.py`)**: Queries `GET /api/ps` and `/api/tags` to inspect loaded and available models. Loads/resplits models via `POST /api/generate` using `num_gpu` (GPU vs. RAM layer split) and `main_gpu` (target GPU), and unloads via `keep_alive: 0`.
   - **llama.cpp (`src/vramctl/backends/llamacpp.py`)**: Detects running `llama-server` / `llama-cli` processes by inspecting `/proc/<pid>/cmdline` and open `.gguf` file descriptors in `/proc/<pid>/fd`. Spawns and reconfigures `llama-server` instances using `CUDA_VISIBLE_DEVICES`, `--n-gpu-layers` (`-ngl`), and `--tensor-split` (`-ts`).
   - **Strata (`src/vramctl/backends/strata.py`)**: **Observe-only.** Strata holds one model at a time and names it on `/v1/status`; identity comes from there, while VRAM and RAM are measured from the driver's compute-app table and process RSS, joined on the pid listening on Strata's port. Loading is refused unless `strata_load_path` is configured, since only `/v1/status` is a confirmed endpoint.
3. **State Aggregation & Re-split Controller (`src/vramctl/manager.py`)**
   Merges hardware PID metrics with backend runtime states and coordinates stop-and-reload workflows when moving or re-splitting a loaded model across GPUs and system RAM.
4. **Interactive TUI (`src/vramctl/tui/app.py`)**
   Built with `textual` for arrow-key navigation across GPUs and active models:
   - **Up / Down / Left / Right**: Navigate between GPU panels and loaded model entries.
   - **`l` (Load)**: Open a modal to pick an available model from Ollama or local model directories, select target GPU(s), and configure the GPU/RAM split.
   - **`s` (Split / Move)**: Reconfigure an active model's GPU assignment or GPU/RAM offload ratio (gracefully unloads and reloads with updated flags).
   - **`u` (Unload)**: Immediately evict the selected model from VRAM/RAM.
   - **`r` (Refresh)**: Manually refresh GPU, RAM, and runtime state, forcing
     a re-read of remote machines rather than using the cached reading.
   - **`k` (Stop process)**: Stop the process behind the selected row, over
     SSH when it is on another machine. Always confirmed first, and the dialog
     opens on Cancel. SIGTERM, then SIGKILL if that is ignored. This is a
     separate key from `u` on purpose: `u` asks a backend to release a model,
     while `k` ends a process. Set `allow_process_kill` to false to remove it.

   Each card also carries a utilization sparkline and, where the card reports
   them, temperature, power against its limit, and fan speed. A sensor the
   card withholds is omitted rather than shown as zero. Samples are only
   appended when a reading is actually fresh, so a cached remote reading does
   not stretch across the whole sparkline.

   VRAM held by something that is not a managed model - a game, ComfyUI, a
   voice server - is listed as an `other` row against the card holding it, so
   a half-full GPU is never unexplained. Those rows carry a `•` rather than a
   spread marker, and `u` and `s` refuse them: vramctl can see that memory but
   not control it. The floor for reporting one is `other_min_mb` (64 MiB by
   default), which keeps a compositor's few MiB out of the list.

## Installation & Usage

```bash
pip install -e .
vramctl
```

### CLI flags

`vramctl` launches the TUI by default. For scripting or a quick look:

```bash
vramctl --once           # print GPU, RAM, and model state, then exit
vramctl --once --json    # same, as JSON
```

Other flags: `--config FILE`, `--ollama-url URL`, `--strata-url URL`,
`--gguf-dir DIR` (repeatable), `--refresh SECONDS`, and
`--no-ollama` / `--no-llamacpp` / `--no-strata` to skip a backend.

### Preferences

Choices made in the UI - the theme and the graph height - are saved to
`$XDG_STATE_HOME/vramctl/prefs.json` (`~/.local/state/vramctl/prefs.json`),
separate from `config.json` so the app never rewrites a hand-written file.
A `theme` or `graph_height` in `config.json` supplies the default; a choice
made in the UI wins, being the more recent explicit one. A missing or
corrupt preferences file falls back to defaults rather than failing to start.

### Configuration

Optional JSON at `~/.config/vramctl/config.json` (or `~/.vramctl.json`).
Without a `nodes` key it manages this machine only, and every other key
mirrors a field of `Config` in `src/vramctl/config.py`.

`nodes` turns it into a cluster view. Each node is a machine; each `ollama`
entry is an endpoint, and `gpu` pins it to one card **on that node**:

```json
{
  "nodes": [
    {
      "name": "InferencePrime",
      "local": true,
      "ollama": [
        {"url": "http://127.0.0.1:11434", "gpu": 0, "label": "george"},
        {"url": "http://127.0.0.1:11435", "gpu": 1, "label": "katie"}
      ],
      "llamacpp": true,
      "strata": {"url": "http://127.0.0.1:8181"}
    },
    {
      "name": "katana-laptop",
      "ssh": "katana-laptop",
      "ollama": [{"url": "http://192.168.157.49:11434", "gpu": 0}],
      "gpus": [{"index": 0, "name": "RTX 4060 Laptop", "vram_mb": 8188}]
    }
  ],
  "gguf_dirs": ["~/models", "/mnt/storage/llm"],
  "llama_server_bin": "llama-server",
  "refresh_seconds": 3.0,
  "remote_telemetry_seconds": 15.0
}
```

Notes on nodes:

- A node is measured locally (`"local": true`) or over SSH (`"ssh": "host"`,
  which must work non-interactively). SSH costs a round trip, so remote GPU
  and RAM readings are cached for `remote_telemetry_seconds`; `r` forces a
  re-read.
- `gpus` declares capacity as a fallback. It is used only when the host cannot
  be measured, and such a card is labelled `declared` with no utilization,
  since nothing was actually read.
- **Pinning `gpu` matters.** When an Ollama instance is confined to one card
  (on InferencePrime, by `DevicePolicy=closed` plus a single `DeviceAllow`),
  that card enumerates as CUDA device 0 *inside* the instance. vramctl then
  sends `main_gpu: 0` and attributes the instance's VRAM to the pinned card
  with no PID correlation. An unpinned endpoint keeps the physical index.
- `llamacpp` and `strata` are **detected** on any node, local or remote: a
  node's process table is read with `ps` over the same SSH call as its GPU and
  RAM figures. Starting and stopping them still needs the process to be local,
  and the on-disk model catalogues are local too.
- A remote probe is a single SSH invocation whose sections are split on a
  sentinel, so a refresh costs one round trip rather than four.
- An endpoint that stops answering is backed off exponentially (5s, doubling
  to a 60s cap) instead of being retried on every tick, so a sleeping laptop
  costs one timeout rather than one per refresh. Inspection uses
  `inspect_timeout` (3s) to stay responsive, while a load gets
  `load_timeout` (600s) because pulling a large model into VRAM is slow.

`VRAMCTL_OLLAMA_URL`, `VRAMCTL_STRATA_URL`, `VRAMCTL_LLAMA_SERVER`,
`VRAMCTL_STRATA_BIN`, and `VRAMCTL_GGUF_DIRS` (`:`-separated) override the
file.
