"""Command-line entry point: scan a directory, write the CSV report."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .analysis import Analysis, Params, analyze
from .media import check_tools, extract
from .report import Cache, make_row, summary, write_csv

DEFAULT_EXT = "mov,mp4,avi,mkv,webm,m4v"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="beatscan", description="Count beats per loop in VJ clips.")
    p.add_argument("directory", type=Path)
    p.add_argument("-o", "--output", type=Path,
                   help="report path (default: ./beatscan_<directory name>.csv)")
    p.add_argument("--ext", default=DEFAULT_EXT, help=f"extensions to scan (default: {DEFAULT_EXT})")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--min-confidence", type=float, default=Params.min_confidence)
    p.add_argument("--tolerance", type=float, default=Params.tolerance,
                   help="score tolerance for alternative beat counts")
    p.add_argument("--bpm-range", default="80-180", help="plausibility window, MIN-MAX")
    p.add_argument("--max-duration", type=float, default=120.0, help="skip clips longer than this (s)")
    p.add_argument("--no-cache", action="store_true", help="ignore the cache and re-analyse everything")
    p.add_argument("--plots", type=Path, help="write one debug plot per clip into this directory")
    args = p.parse_args(argv)
    try:
        lo, hi = (float(v) for v in args.bpm_range.split("-"))
        assert 0 < lo < hi
    except (ValueError, AssertionError):
        p.error("--bpm-range must look like 80-180")
    args.bpm = (lo, hi)
    if not args.directory.is_dir():
        p.error(f"not a directory: {args.directory}")
    return args


def find_clips(root: Path, exts: str) -> list[Path]:
    wanted = {"." + e.strip().lower().lstrip(".") for e in exts.split(",") if e.strip()}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in wanted and not p.name.startswith("._")
    )


class Progress:
    def __init__(self, total: int):
        self.total, self.done, self.tty = total, 0, sys.stderr.isatty()

    def step(self, label: str) -> None:
        self.done += 1
        msg = f"[{self.done}/{self.total}] {label}"
        if self.tty:
            sys.stderr.write("\r\033[K" + msg[:120])
            sys.stderr.flush()
        elif self.done % 25 == 0 or self.done == self.total:
            print(msg, file=sys.stderr)

    def log(self, msg: str) -> None:
        if self.tty:
            sys.stderr.write("\r\033[K")
        print(msg, file=sys.stderr)

    def end(self) -> None:
        if self.tty:
            sys.stderr.write("\r\033[K")


def main(argv=None) -> int:
    args = parse_args(argv)
    check_tools()
    root = args.directory.resolve()
    clips = find_clips(root, args.ext)
    if not clips:
        print(f"No clips with extensions {args.ext} under {root}", file=sys.stderr)
        return 1

    output = (args.output or Path(f"beatscan_{root.name}.csv")).resolve()
    cache = Cache(output.with_name(output.stem + ".cache.json"), enabled=not args.no_cache)
    params = Params(min_confidence=args.min_confidence, tolerance=args.tolerance,
                    bpm_min=args.bpm[0], bpm_max=args.bpm[1])

    extracted: dict[Path, dict] = {}
    todo = []
    for c in clips:
        hit = cache.get(c)
        if hit is not None:
            extracted[c] = hit
        else:
            todo.append(c)
    print(f"{len(clips)} clips, {len(clips) - len(todo)} cached, analysing {len(todo)} "
          f"with {args.workers} workers", file=sys.stderr)

    progress = Progress(len(clips))
    for c in clips:
        if c in extracted:
            progress.step(f"cached  {c.relative_to(root)}")
    if todo:
        last_save = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract, str(c), args.max_duration): c for c in todo}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    data = fut.result()
                except Exception as e:  # worker crashed
                    data = {"error": f"worker failed: {e}", "warnings": []}
                extracted[c] = data
                cache.put(c, data)
                for w in data.get("warnings", []):
                    progress.log(f"warning: {c.relative_to(root)}: {w}")
                progress.step(str(c.relative_to(root)))
                if time.monotonic() - last_save > 30:
                    cache.save()
                    last_save = time.monotonic()
    progress.end()
    cache.save()

    rows = []
    for c in clips:
        rel = str(c.relative_to(root))
        data = extracted[c]
        probe = data.get("probe")
        if "error" in data:
            rows.append(make_row(rel, Analysis("error"), probe, None, data["error"]))
            continue
        if "skipped" in data:
            rows.append(make_row(rel, Analysis("skipped", reason=data["skipped"]), probe, probe.get("nb_frames")))
            continue
        frames = len(data["motion"])
        if frames / probe["fps"] > args.max_duration:  # cached, but now over the limit
            rows.append(make_row(rel, Analysis("skipped", reason=f"longer than {args.max_duration:g}s"),
                                 probe, frames))
            continue
        try:
            a = analyze(np.array(data["motion"]), np.array(data["brightness"]), probe["fps"], params)
        except Exception as e:
            rows.append(make_row(rel, Analysis("error"), probe, frames, f"analysis failed: {e}"))
            continue
        rows.append(make_row(rel, a, probe, frames))
        if args.plots:
            from .plots import write_plot
            name = rel.replace(os.sep, "__") + ".png"
            try:
                write_plot(args.plots / name, rel, a, data["motion"], data["brightness"], probe["fps"])
            except Exception as e:
                print(f"warning: plot failed for {rel}: {e}", file=sys.stderr)

    write_csv(output, rows)
    print(summary(rows))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
