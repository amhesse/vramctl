"""Preferences the app itself writes, persisted between sessions.

Kept in a separate file from ``config.json``: that one is hand-written (and
may be managed by something else), so the app never rewrites it and risks
losing a comment or a field it does not know about. This file is the app's
own state, and it follows the XDG state directory rather than the config
directory because it records what the user last did, not how they set things
up.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Prefs:
    """What the UI remembers. Every field needs a sane default.

    A preference that cannot be honoured - a theme that no longer exists, say
    - is ignored at the point of use rather than repaired here, so a future
    version of Textual renaming a theme cannot stop the app starting.
    """

    theme: str = ""
    # 0 means "follow the config default" rather than a height of zero.
    graph_height: int = 0


def prefs_path() -> Path:
    """``$XDG_STATE_HOME/vramctl/prefs.json``, honouring the env var."""
    base = os.environ.get("XDG_STATE_HOME") or "~/.local/state"
    return Path(base).expanduser() / "vramctl" / "prefs.json"


def load_prefs(path: Path | None = None) -> Prefs:
    """Read saved preferences, falling back to defaults on any problem.

    Preferences are a convenience: a missing, truncated or hand-mangled file
    must never stop the app from starting, so everything here degrades to the
    default rather than raising.
    """
    file = path or prefs_path()
    try:
        raw = json.loads(file.read_text())
    except (OSError, json.JSONDecodeError):
        return Prefs()
    if not isinstance(raw, dict):
        return Prefs()

    known = {f.name: f for f in fields(Prefs)}  # noqa: F841 - shape check below
    prefs = Prefs()
    for key, value in raw.items():
        field = known.get(key)
        if field is None:
            continue
        # Ignore anything of the wrong shape instead of trusting the file.
        default = getattr(Prefs(), key)
        if isinstance(default, bool) or not isinstance(value, type(default)):
            continue
        setattr(prefs, key, value)
    return prefs


def save_prefs(prefs: Prefs, path: Path | None = None) -> bool:
    """Write preferences atomically. Returns False if they could not be saved.

    The write goes to a temporary file in the same directory and is then
    renamed over the target, so an interrupted save cannot leave a half-written
    file that the next start would reject.
    """
    file = path or prefs_path()
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=file.parent,
            prefix=file.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(asdict(prefs), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp = Path(handle.name)
        os.replace(temp, file)
        return True
    except OSError:
        try:
            temp.unlink(missing_ok=True)
        except (OSError, NameError, UnboundLocalError):
            pass
        return False
