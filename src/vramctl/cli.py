"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

from vramctl import __version__
from vramctl.config import load_config
from vramctl.manager import ClusterManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vramctl",
        description="Inspect and split LLM workloads across NVIDIA GPUs and system RAM.",
    )
    parser.add_argument("--version", action="version", version=f"vramctl {__version__}")
    parser.add_argument("-c", "--config", help="path to a JSON config file")
    parser.add_argument("--ollama-url", help="override the Ollama base URL")
    parser.add_argument("--strata-url", help="override the Strata base URL")
    parser.add_argument(
        "--gguf-dir",
        action="append",
        dest="gguf_dirs",
        metavar="DIR",
        help="directory to scan for .gguf models (repeatable)",
    )
    parser.add_argument(
        "--no-ollama", action="store_true", help="skip the Ollama backend"
    )
    parser.add_argument(
        "--no-llamacpp", action="store_true", help="skip the llama.cpp backend"
    )
    parser.add_argument(
        "--no-strata", action="store_true", help="skip the Strata backend"
    )
    parser.add_argument(
        "--refresh",
        type=float,
        metavar="SECONDS",
        help="auto-refresh interval (default 3)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print current GPU/model state and exit, without starting the TUI",
    )
    parser.add_argument(
        "--json", action="store_true", help="with --once, emit JSON instead of text"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"vramctl: {exc}", file=sys.stderr)
        return 2

    if args.ollama_url:
        config.ollama_url = args.ollama_url.rstrip("/")
    if args.strata_url:
        config.strata_url = args.strata_url.rstrip("/")
    if args.gguf_dirs:
        config.gguf_dirs = args.gguf_dirs
    if args.refresh is not None:
        config.refresh_seconds = max(0.5, args.refresh)
    config.ollama_enabled = not args.no_ollama
    config.llamacpp_enabled = not args.no_llamacpp
    config.strata_enabled = not args.no_strata

    manager = ClusterManager(config)

    if args.once:
        return _print_once(manager, as_json=args.json)

    from vramctl.tui.app import VramctlApp

    VramctlApp(manager).run()
    return 0


def _print_once(manager: ClusterManager, as_json: bool) -> int:
    state = manager.refresh_state()

    if as_json:
        payload = {
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_total_mb": g.memory_total_mb,
                    "memory_used_mb": g.memory_used_mb,
                    "utilization_pct": g.utilization_pct,
                    "pids": g.pids,
                }
                for g in state.gpus
            ],
            "ram": vars(state.ram) if state.ram else None,
            "models": [
                {
                    "backend": m.backend,
                    "name": m.name,
                    "pid": m.pid,
                    "vram_mb_per_gpu": m.vram_mb_per_gpu,
                    "ram_mb": m.ram_mb,
                    "size_mb": m.size_mb,
                    "placement": m.placement.describe(),
                }
                for m in state.models
            ],
            "errors": state.errors,
        }
        print(json.dumps(payload, indent=2))
        return 1 if state.errors and not state.gpus else 0

    for gpu in state.gpus:
        print(
            f"GPU {gpu.index}  {gpu.short_name:<24} "
            f"{gpu.memory_used_mb:>6}/{gpu.memory_total_mb} MiB  "
            f"{gpu.utilization_pct:>3}% util"
        )
    if state.ram:
        ram = state.ram
        print(f"RAM        {ram.used_mb:>6}/{ram.total_mb} MiB")

    print()
    if state.models:
        for model in state.models:
            where = model.placement.describe()
            print(
                f"[{model.backend:<8}] {model.name:<36} "
                f"VRAM {model.vram_mb:>6} MiB  RAM {model.ram_mb:>6} MiB  {where}"
            )
    else:
        print("No models resident.")

    for error in state.errors:
        print(f"warning: {error}", file=sys.stderr)

    return 1 if state.errors and not state.gpus else 0


if __name__ == "__main__":
    raise SystemExit(main())
