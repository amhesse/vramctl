"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

import shutil
from pathlib import Path

import httpx

from vramctl import __version__
from vramctl.config import CONFIG_PATHS, EndpointConfig, load_config
from vramctl.manager import ClusterManager
from vramctl.nvidia import NvidiaError, query_gpus


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
        "--init",
        action="store_true",
        help="probe this machine and print a starter config to stdout",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="with --init, save the config instead of printing it",
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


def _apply_endpoint_overrides(config, args) -> None:
    """Let --ollama-url and --strata-url override whatever the config built.

    An explicit URL on the command line replaces every Ollama endpoint: it is
    a deliberate "look here instead", which is more useful for a one-off check
    than being merged with the configured set.
    """
    if args.ollama_url:
        replacement = EndpointConfig(url=args.ollama_url.rstrip("/"))
        for node in config.nodes:
            node.endpoints = [replacement] if node.local else []
    if args.strata_url:
        for node in config.nodes:
            if node.local:
                node.strata_url = args.strata_url.rstrip("/")


def _discover(config) -> tuple[dict, list[str]]:
    """Probe this machine for GPUs, Ollama endpoints and local models.

    Only loopback is probed, and only the handful of ports Ollama is normally
    run on, so this does not scan anyone's network.
    """
    import socket

    notes: list[str] = []
    gpus = []
    try:
        gpus = query_gpus()
        notes.append(f"found {len(gpus)} GPU(s): " + ", ".join(
            f"{g.index}:{g.short_name}" for g in gpus
        ))
    except NvidiaError as exc:
        notes.append(f"no GPUs readable ({exc})")

    endpoints: list[dict] = []
    for port in range(11434, 11440):
        url = f"http://127.0.0.1:{port}"
        try:
            resp = httpx.get(f"{url}/api/version", timeout=1.0)
            resp.raise_for_status()
            version = (resp.json() or {}).get("version", "?")
        except (httpx.HTTPError, ValueError):
            continue
        entry: dict = {"url": url}
        # With a single card there is no ambiguity. With several, only the
        # operator knows which instance is pinned where.
        if len(gpus) == 1:
            entry["gpu"] = gpus[0].index
        endpoints.append(entry)
        notes.append(f"Ollama {version} at {url}")
    if not endpoints:
        notes.append("no Ollama found on 11434-11439; add its url by hand")
    elif len(endpoints) > 1 and len(gpus) > 1:
        notes.append(
            f"{len(endpoints)} endpoints and {len(gpus)} GPUs: add \"gpu\": <index> "
            "to each endpoint that is pinned to one card, or leave it out if "
            "an instance can see them all"
        )

    binary = shutil.which(config.llama_server_bin)
    if binary:
        notes.append(f"llama-server at {binary}")
    else:
        notes.append(
            f"no {config.llama_server_bin} on PATH; set llama_server_bin to "
            "its full path to manage llama.cpp"
        )

    model_dirs = []
    for raw in config.gguf_dirs:
        directory = Path(raw).expanduser()
        if not directory.is_dir():
            continue
        count = sum(1 for _ in directory.rglob("*.gguf"))
        if count:
            model_dirs.append(str(directory))
            notes.append(f"{count} .gguf under {directory}")

    node: dict = {
        "name": socket.gethostname() or "local",
        "local": True,
        "ollama": endpoints,
        "llamacpp": bool(binary),
    }
    out: dict = {"nodes": [node]}
    if model_dirs:
        out["gguf_dirs"] = model_dirs
    if binary and binary != config.llama_server_bin:
        out["llama_server_bin"] = binary
    return out, notes


def _print_init(config, write: bool) -> int:
    discovered, notes = _discover(config)
    for note in notes:
        print(f"# {note}", file=sys.stderr)

    body = json.dumps(discovered, indent=2) + "\n"
    if not write:
        print(body, end="")
        print(
            "\n# redirect this into ~/.config/vramctl/config.json, "
            "or re-run with --write",
            file=sys.stderr,
        )
        return 0

    target = Path(CONFIG_PATHS[0]).expanduser()
    if target.exists():
        print(
            f"vramctl: {target} already exists; not overwriting it",
            file=sys.stderr,
        )
        return 2
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    except OSError as exc:
        print(f"vramctl: cannot write {target}: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {target}", file=sys.stderr)
    return 0


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
    # Nodes are built while the config is read, so a flag that changes an
    # endpoint has to be pushed into them afterwards or it is silently
    # ignored.
    _apply_endpoint_overrides(config, args)
    if args.gguf_dirs:
        config.gguf_dirs = args.gguf_dirs
    if args.refresh is not None:
        config.refresh_seconds = max(0.5, args.refresh)
    config.ollama_enabled = not args.no_ollama
    config.llamacpp_enabled = not args.no_llamacpp
    config.strata_enabled = not args.no_strata

    if args.init:
        return _print_init(config, args.write)

    manager = ClusterManager(config)

    if args.once:
        return _print_once(manager, as_json=args.json)

    from vramctl.tui.app import VramctlApp

    VramctlApp(manager).run()
    return 0


def _print_once(manager: ClusterManager, as_json: bool) -> int:
    state = manager.refresh_state(force_telemetry=True)

    if as_json:
        payload = {
            "nodes": [
                {
                    "name": n.name,
                    "reachable": n.reachable,
                    "error": n.error or None,
                    "ram": vars(n.ram) if n.ram else None,
                    "gpus": [
                        {
                            "index": g.index,
                            "name": g.name,
                            "memory_total_mb": g.memory_total_mb,
                            "memory_used_mb": g.memory_used_mb,
                            "utilization_pct": g.utilization_pct,
                            "temperature_c": g.temperature_c,
                            "power_w": g.power_w,
                            "power_limit_w": g.power_limit_w,
                            "fan_pct": g.fan_pct,
                            "declared": g.declared,
                            "pids": g.pids,
                        }
                        for g in n.gpus
                    ],
                }
                for n in state.nodes
            ],
            "models": [
                {
                    "node": m.node,
                    "backend": m.backend,
                    "endpoint": m.endpoint,
                    "name": m.name,
                    "pid": m.pid,
                    "vram_mb_per_gpu": m.vram_mb_per_gpu,
                    "vram_mb": m.vram_mb,
                    "ram_mb": m.ram_mb,
                    "size_mb": m.size_mb,
                    "spread": m.spread,
                    "placement": m.placement_label,
                    "manageable": m.manageable,
                }
                for m in state.models
            ],
            "errors": state.errors,
        }
        print(json.dumps(payload, indent=2))
        return 1 if state.errors and not state.gpus else 0

    for node in state.nodes:
        header = node.name
        if not node.reachable:
            header += "  (unreachable, declared capacity)"
        print(header)
        for gpu in node.gpus:
            if gpu.declared:
                held = sum(
                    m.vram_mb_per_gpu.get(gpu.index, 0)
                    for m in state.models_on_gpu(node.name, gpu.index)
                )
                print(
                    f"  GPU {gpu.index}  {gpu.short_name:<24} "
                    f"{held:>6}/{gpu.memory_total_mb} MiB held/declared"
                )
            else:
                sensors = f"  {gpu.sensors}" if gpu.sensors else ""
                print(
                    f"  GPU {gpu.index}  {gpu.short_name:<24} "
                    f"{gpu.memory_used_mb:>6}/{gpu.memory_total_mb} MiB  "
                    f"{gpu.utilization_pct:>3}% util{sensors}"
                )
        if node.ram:
            print(f"  RAM        {node.ram.used_mb:>6}/{node.ram.total_mb} MiB")
        print()

    if state.models:
        print(f"{'MACHINE':<16} {'BACKEND':<9} {'MODEL':<34} "
              f"{'VRAM':>8} {'RAM':>8}  PLACEMENT")
        for m in state.models:
            print(
                f"{m.node:<16} {m.backend:<9} {m.name[:34]:<34} "
                f"{m.vram_mb:>8} {m.ram_mb:>8}  {m.placement_label}"
            )
    else:
        print("No models resident.")

    for error in state.errors:
        print(f"warning: {error}", file=sys.stderr)

    return 1 if state.errors and not state.gpus else 0


if __name__ == "__main__":
    raise SystemExit(main())
