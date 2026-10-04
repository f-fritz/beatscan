"""Cache, CSV report and terminal summary."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

from .analysis import Analysis

CACHE_VERSION = 1

COLUMNS = [
    "path", "status", "beats", "confidence", "alt_beats", "character", "signal",
    "whole_loop", "loop_seamless", "duration_s", "fps", "frames", "implied_bpm",
    "codec", "error",
]
STATUS_ORDER = ["rhythmic", "rhythmic_partial", "non_rhythmic", "static", "skipped", "error"]


class Cache:
    """Decoded signals keyed by path, size and mtime.

    Signals are cached rather than results, so changing analysis options
    re-uses the expensive decode.
    """

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.entries: dict = {}
        if enabled and path.exists():
            try:
                data = json.loads(path.read_text())
                if data.get("version") == CACHE_VERSION:
                    self.entries = data.get("entries", {})
            except (OSError, ValueError):
                pass

    @staticmethod
    def _stamp(file: Path) -> tuple[int, float]:
        st = file.stat()
        return st.st_size, st.st_mtime

    def get(self, file: Path) -> dict | None:
        e = self.entries.get(str(file))
        if e is None:
            return None
        size, mtime = self._stamp(file)
        if e.get("size") != size or e.get("mtime") != mtime:
            return None
        return e["data"]

    def put(self, file: Path, data: dict) -> None:
        # Errors and length-based skips are not cached: retry next time.
        if "motion" not in data:
            return
        size, mtime = self._stamp(file)
        self.entries[str(file)] = {"size": size, "mtime": mtime, "data": data}

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "entries": self.entries}))
        os.replace(tmp, self.path)


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v.is_integer() and digits == 0:
            return str(int(v))
        return f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:g}"
    return str(v)


def make_row(rel: str, a: Analysis, probe: dict | None, frames: int | None, error: str = "") -> dict:
    fps = (probe or {}).get("fps")
    duration = frames / fps if frames and fps else (probe or {}).get("container_duration")
    beats = a.beats
    if beats is not None and a.status == "rhythmic":
        beats = int(beats)
    return {
        "path": rel,
        "status": a.status,
        "beats": _fmt(beats, 2),
        "confidence": _fmt(round(a.confidence, 3)) if a.status not in ("static", "skipped", "error") else "",
        "alt_beats": ";".join(str(k) for k in a.alt_beats),
        "character": a.character,
        "signal": a.signal,
        "whole_loop": _fmt(a.whole_loop),
        "loop_seamless": _fmt(a.loop_seamless),
        "duration_s": _fmt(duration, 3),
        "fps": _fmt(fps, 3),
        "frames": _fmt(frames),
        "implied_bpm": _fmt(a.implied_bpm, 1),
        "codec": (probe or {}).get("codec", ""),
        "error": error or a.reason,
    }


def sort_rows(rows: list[dict]) -> list[dict]:
    def key(r):
        conf = float(r["confidence"]) if r["confidence"] else -1.0
        return (STATUS_ORDER.index(r["status"]), -conf, r["path"])
    return sorted(rows, key=key)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sort_rows(rows))


def summary(rows: list[dict]) -> str:
    status = Counter(r["status"] for r in rows)
    beats = Counter(r["beats"] for r in rows if r["status"] == "rhythmic")
    codecs = Counter(r["codec"] or "?" for r in rows)
    failed_codecs = Counter(r["codec"] or "?" for r in rows if r["status"] == "error")
    lines = [f"{len(rows)} clips"]
    lines.append("  status:  " + ", ".join(f"{s} {status[s]}" for s in STATUS_ORDER if status[s]))
    if beats:
        lines.append("  beats:   " + ", ".join(
            f"{b}: {beats[b]}" for b in sorted(beats, key=lambda b: float(b))))
    lines.append("  codecs:  " + ", ".join(f"{c} {n}" for c, n in codecs.most_common()))
    if failed_codecs:
        lines.append("  errors by codec: " + ", ".join(f"{c} {n}" for c, n in failed_codecs.most_common()))
    return "\n".join(lines)
