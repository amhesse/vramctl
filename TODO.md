# TODO Checklist

- [x] Write project overview, runtime architecture, and keybinding documentation in `README.md`.
- [x] Create `pyproject.toml` with package metadata, dependencies (`textual`, `httpx`, `psutil`), and the `vramctl = "vramctl.cli:main"` CLI entrypoint.
- [x] Create `src/vramctl/__init__.py` and `src/vramctl/config.py` defining `GPUInfo`, `LoadedModel`, `PlacementConfig`, and `load_config()` for Ollama URL, Strata settings, and local model scan directories.
- [x] Implement `query_gpus()` and `query_system_ram()` in `src/vramctl/nvidia.py` to parse `nvidia-smi` per-GPU VRAM and compute PID allocations alongside host RAM stats from `psutil.virtual_memory()`.
- [x] Implement the Ollama backend adapter in `src/vramctl/backends/ollama.py` (`list_running()`, `list_available()`, `load_model(name, num_gpu, main_gpu)`, and `unload_model(name)`).
- [x] Implement the `llama.cpp` backend adapter in `src/vramctl/backends/llamacpp.py` (`detect_processes(gpu_pids)`, `discover_gguf_models(search_dirs)`, `spawn_model(gguf_path, gpu_ids, n_gpu_layers, tensor_split, port)`, and `stop_model(pid)`).
- [x] Implement the Strata backend adapter in `src/vramctl/backends/strata.py` (`detect_models(gpu_pids)`, `discover_strata_models(search_dirs)`, `load_model(model_id, gpu_ids, ram_offload)`, and `unload_model(target)`).
- [x] Implement `ClusterManager` in `src/vramctl/manager.py` with `refresh_state()` (correlating GPU PIDs with Ollama, `llama.cpp`, and Strata instances) and `resplit_model(model, new_placement)` (graceful unload and reload with updated GPU/RAM split parameters).
- [x] Implement the Textual TUI in `src/vramctl/tui/app.py` (`GpuCard`, `ModelRow`, `LoadOrSplitModal`, and `VramctlApp`) with arrow-key navigation and `l` (load), `u` (unload), `s` (split/move), and `r` (refresh) keybindings.
- [x] Implement `main()` in `src/vramctl/cli.py` to parse CLI flags and launch `VramctlApp`.

## Next

- [ ] Multi-endpoint `nodes[]` config: one Ollama URL per GPU, with the endpoint
      as the GPU identity (`:11434` -> 3090 Ti, `:11435` -> 4070 Ti SUPER, both
      enforced by `DevicePolicy=closed` + a single `DeviceAllow` in each unit's
      `zz-gpu-pin.conf`).
- [ ] Fix `OllamaBackend.load()` to send `main_gpu: 0` per endpoint. A physical
      index is wrong: each instance's cgroup exposes exactly one card, which
      enumerates as device 0 inside the service.
- [ ] Drop the `_attribute_vram()` PID heuristic for endpoint-pinned instances;
      the endpoint already identifies the GPU exactly.
- [ ] SSH telemetry for `katana-laptop` (192.168.157.49, passwordless, user
      `aaron`), with declared VRAM capacity as the fallback. Short timeout and
      graceful `unreachable` state, since the laptop sleeps.
- [ ] Blocked, needs Aaron: laptop `nvidia-smi` returns `Failed to initialize
      NVML: Driver/library version mismatch` (NVML 615.78). Needs a reboot.
- [ ] Blocked, needs Aaron: Strata's real port (the `8181` default is invented)
      and its install source - no pacman/pip/docker/dotfile trace remains.
- [ ] Optional: `vramctl split <model> --gpu N` as a non-interactive subcommand.
      Only the TUI `s` key does this today.
