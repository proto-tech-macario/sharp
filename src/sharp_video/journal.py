"""Per-run processing journal: completed frames, failed frames, errors (spec §32).

Stored as JSON next to the per-frame cache and rewritten atomically after
every update, so a crash never leaves it half-written. Thread-safe.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class Journal:
    """Records which frames finished and which failed (with the error)."""

    def __init__(self, path: str | Path):
        """Open the journal at `path`, loading it if it already exists."""
        self.path = Path(path)
        self._lock = threading.Lock()
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        else:
            self._data = {"frames": {}, "last_completed": None}

    def mark_done(self, index: int, pts: int, timestamp: float, in_order: bool = True) -> None:
        """Record frame `index` as done.

        `in_order` means every earlier selected frame is done too, which is
        what `last_completed` (the resume point) tracks.
        """
        with self._lock:
            self._data["frames"][str(index)] = {
                "status": "done", "pts": pts, "timestamp": timestamp, "error": None,
            }
            if in_order:
                previous = self._data["last_completed"]
                self._data["last_completed"] = index if previous is None else max(previous, index)
            self._save()

    def mark_failed(self, index: int, pts: int, timestamp: float, error: str) -> None:
        """Record frame `index` as failed, with its error text."""
        with self._lock:
            self._data["frames"][str(index)] = {
                "status": "failed", "pts": pts, "timestamp": timestamp, "error": error,
            }
            self._save()

    def is_done(self, index: int) -> bool:
        """Whether frame `index` has completed."""
        with self._lock:
            entry = self._data["frames"].get(str(index))
            return entry is not None and entry["status"] == "done"

    @property
    def last_completed(self) -> int | None:
        """Last frame completed with every earlier frame done: the resume point."""
        return self._data["last_completed"]

    @property
    def failures(self) -> list[dict[str, Any]]:
        """Every failed frame with its PTS, timestamp and error, in index order."""
        with self._lock:
            return [
                {"index": int(key), "pts": entry["pts"], "timestamp": entry["timestamp"],
                 "error": entry["error"]}
                for key, entry in sorted(self._data["frames"].items(), key=lambda kv: int(kv[0]))
                if entry["status"] == "failed"
            ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)
