"""Generate a Jupyter notebook that walks through the analysis of one clip.

    beatscan-explain clip.mov [-o notebook.ipynb] [--no-execute]

The notebook repeats each step of `analysis.analyze` in its own cell, with
plots, and finishes by checking that the step-by-step result matches the
scanner's. Executing it needs `nbclient` and `ipykernel`
(`pip install -e '.[notebook]'`); without them the notebook is written
unexecuted, ready to run in Jupyter or VS Code.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .analysis import Params
from .media import check_tools, probe


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip()}


def build_notebook(clip: Path, params: Params) -> dict:
    cells = [
        _md(f"""
            # How beatscan analyses `{clip.name}`

            Each cell below runs one step of the analysis on this clip. Change
            `PARAMS` and re-run all cells to see how the options affect the
            result.
        """),
        _code(f"""
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            from beatscan import analysis as A
            from beatscan.media import decode_frames, probe

            CLIP = Path({str(clip)!r})
            PARAMS = A.Params(min_confidence={params.min_confidence}, tolerance={params.tolerance},
                              bpm_min={params.bpm_min}, bpm_max={params.bpm_max}, acf_min={params.acf_min})
            plt.rcParams.update({{"figure.figsize": (11, 3.2), "axes.grid": True, "grid.alpha": 0.3}})
        """),

        _md("""
            ## 1. Probe

            `ffprobe` reports codec, pixel format and frame rate. Clips whose pixel
            format has an alpha channel are composited over black before analysis.
        """),
        _code("""
            pr = probe(str(CLIP))
            for key, value in pr.to_dict().items():
                print(f"{key:20s} {value}")
            print(f"{'has_alpha':20s} {pr.has_alpha}")
            if pr.vfr:
                print("\\nVariable frame rate: frames are treated as evenly spaced.")
        """),

        _md("""
            ## 2. Decode

            ffmpeg decodes every frame at the native frame rate and scales it to
            **64×36 grey**, which is all the analysis looks at. Below, a contact
            sheet of evenly spaced frames: the larger preview is for you, the
            small one is the analyser's input.
        """),
        _code("""
            tiny = np.stack(list(decode_frames(str(CLIP), pr.has_alpha)))
            preview = np.stack(list(decode_frames(str(CLIP), pr.has_alpha, 192, 108)))
            n = len(tiny)
            fps = pr.fps
            duration = n / fps
            print(f"decoded {n} frames at {fps:.3f} fps -> duration {duration:.3f} s "
                  f"(probe reported {pr.nb_frames} frames)")

            picks = np.linspace(0, n - 1, 8).round().astype(int)
            fig, axes = plt.subplots(2, len(picks), figsize=(14, 3.4))
            for col, i in enumerate(picks):
                axes[0, col].imshow(preview[i], cmap="gray", vmin=0, vmax=255)
                axes[0, col].set_title(f"frame {i}\\n{i / fps:.2f} s", fontsize=8)
                axes[1, col].imshow(tiny[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            axes[0, 0].set_ylabel("preview"); axes[1, 0].set_ylabel("64×36 input")
            plt.tight_layout(); plt.show()
        """),

        _md("""
            ## 3. Per-frame signals

            * **motion**: mean absolute difference between a frame and the one
              before it. The clip is a loop, so frame 0 is compared with the last
              frame.
            * **brightness**: mean grey level of each frame.

            If the last→first jump is much bigger than any jump inside the clip,
            the loop is not seamless. That jump is then replaced by the median, so
            the seam doesn't count as a beat.
        """),
        _code("""
            motion, brightness = A.frame_signals(tiny)
            t = np.arange(n) / fps

            seamless = A.loop_seamless(motion)
            limit = motion[1:].max() * 1.1 + 0.5
            print(f"seam (last->first) difference {motion[0]:.2f}, limit {limit:.2f} -> seamless: {seamless}")
            m = motion.copy()
            if not seamless:
                m[0] = np.median(m[1:])
                print("seam jump replaced by the median for the analysis")

            fig, ax = plt.subplots()
            ax.plot(t, m, lw=1, label="motion")
            ax.set_xlabel("seconds"); ax.set_ylabel("motion (grey levels)")
            ax2 = ax.twinx()
            ax2.plot(t, brightness, lw=1, color="tab:orange", label="brightness")
            ax2.set_ylabel("brightness"); ax2.grid(False)
            fig.legend(loc="upper right"); plt.show()

            flat_m, flat_b = m.std() < A.FLAT_STD, brightness.std() < A.FLAT_STD
            print(f"motion:     mean {m.mean():.2f}, std {m.std():.3f}" + ("  (flat, ignored)" if flat_m else ""))
            print(f"brightness: mean {brightness.mean():.2f}, std {brightness.std():.3f}" + ("  (flat, ignored)" if flat_b else ""))
            if flat_m and flat_b:
                verdict = "static" if motion[1:].mean() < A.STATIC_MOTION else "non_rhythmic (constant motion)"
                print(f"\\nBoth signals are flat -> {verdict}. The remaining steps have nothing to work with.")
        """),

        _md("""
            ## 4. Cycles per clip

            Each signal is centred and scaled to unit variance, then transformed
            with an FFT over the **whole clip**. Bin *k* then means "*k* cycles per
            clip", so it is directly a candidate beat count.

            Each *k* is scored by the energy at *k*, 2*k*, 3*k* and 4*k* (weighted
            1, ½, ⅓, ¼), **minus** half the energy that sits between those
            harmonics. The penalty stops 16 from winning on an 8-pulse clip,
            because 16 would leave energy at 8, 24, …

            **Confidence** is the share of energy on multiples of *k*, corrected
            for what random noise would put there. 0 means noise, 1 means an exact
            repeat. Orange bars below are the multiples of the winning *k*.
        """),
        _code("""
            signals = {}
            if not flat_m:
                signals["motion"] = A.analyse_signal("motion", A.normalise(m))
            if not flat_b:
                signals["brightness"] = A.analyse_signal("brightness", A.normalise(brightness))

            for name, s in signals.items():
                share = s.power / s.power[1:].sum()
                kmax = len(s.scores) - 1
                xmax = min(len(share) - 1, max(40, s.k * 5))
                fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 3.2))
                ks = np.arange(1, xmax + 1)
                colours = ["tab:orange" if s.k and k % s.k == 0 else "tab:blue" for k in ks]
                a1.bar(ks, share[1:xmax + 1], color=colours)
                a1.set_title(f"{name}: share of energy per cycle count (orange = multiples of k={s.k})", fontsize=9)
                a1.set_xlabel("cycles per clip (k)")
                a2.plot(np.arange(len(s.scores)), s.scores, marker=".", lw=1)
                a2.axvline(s.k, color="red", ls="--", label=f"best k={s.k}")
                a2.set_xlim(0, min(kmax, xmax)); a2.legend()
                a2.set_title(f"{name}: harmonic score (k up to n/4 = {kmax})", fontsize=9)
                a2.set_xlabel("cycles per clip (k)")
                plt.tight_layout(); plt.show()

                top = np.argsort(np.nan_to_num(s.scores, nan=-np.inf))[::-1][:5]
                print(f"{name}: top scores")
                for k in top:
                    print(f"   k={k:3d}  score {s.scores[k]:+.3f}  confidence {A.lattice_confidence(s.power, k):.3f}"
                          f"  ({k * 60 / duration:.1f} BPM if one beat per cycle)")
                print()

            best = max(signals.values(), key=lambda s: s.confidence) if signals else None
            if best:
                print(f"stronger signal: {best.name}, k={best.k}, confidence {best.confidence:.3f} "
                      f"(threshold {PARAMS.min_confidence})")
        """),

        _md("""
            ## 5. From cycles to beats

            The signal alone cannot tell 8 beats from 4 beats with a pulse every
            half beat, or from 16 beats with a pulse every other beat. Candidates
            are:

            * every *k* scoring within `tolerance` of the best, and
            * the best *k* × 2ⁿ (halved and doubled).

            Candidates whose tempo (`k × 60 / duration`) falls outside the BPM
            window are dropped, unless none fall inside. The rest are ranked:
            powers of two first, then multiples of 4, then multiples of 3, then
            the one closest to the detected *k*.
        """),
        _code("""
            CLASS = {0: "power of 2", 1: "multiple of 4", 2: "multiple of 3", 3: "other"}
            if best and best.confidence >= PARAMS.min_confidence:
                beats, alts, cands = A.resolve_beats(best.k, best.scores, duration, PARAMS)
                print(f"{'beats':>6} {'score':>8} {'BPM':>7}  {'in window':9}  class")
                for c in cands:
                    inside = PARAMS.bpm_min <= c.bpm <= PARAMS.bpm_max
                    mark = "  <- chosen" if c.beats == beats else ("  (alternative)" if c.beats in alts else "")
                    score = "" if np.isnan(c.score) else f"{c.score:+.3f}"
                    print(f"{c.beats:6d} {score:>8} {c.bpm:7.1f}  {str(inside):9}  {CLASS[A.pref_class(c.beats)]}{mark}")
                print(f"\\n-> {beats} beats at {beats * 60 / duration:.1f} BPM; alternatives: {alts or 'none'}")

                fig, ax = plt.subplots()
                ax.plot(t, m, lw=1, label="motion")
                ax.plot(t, (brightness - brightness.min()) / (np.ptp(brightness) or 1) * m.max(),
                        lw=1, alpha=0.6, label="brightness (scaled)")
                for b in np.arange(beats) * duration / beats:
                    ax.axvline(b, color="grey", lw=0.8, alpha=0.6)
                ax.set_title(f"beat grid: {beats} beats", fontsize=9); ax.set_xlabel("seconds"); ax.legend()
                plt.show()
            else:
                beats = None
                print("Confidence below the threshold: no whole-loop beat count. See the fallback below.")
        """),
        _md("""
            ### Frames on the beat

            The first frame of each beat (up to 16). On a correct result,
            these frames should look alike, or show the same phase of the
            movement.
        """),
        _code("""
            if beats:
                shown = min(beats, 16)
                idx = (np.arange(shown) * n / beats).round().astype(int)
                cols = min(shown, 8)
                rows = int(np.ceil(shown / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(14, 1.9 * rows), squeeze=False)
                for ax in axes.flat:
                    ax.set_axis_off()
                for j, i in enumerate(idx):
                    ax = axes.flat[j]
                    ax.imshow(preview[i], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"beat {j + 1} (frame {i})", fontsize=8)
                plt.tight_layout(); plt.show()
            else:
                print("no beat count to show")
        """),

        _md("""
            ## 6. Fallback: autocorrelation

            This step only matters when the spectrum is not confident. That
            happens when the clip is periodic but not cut to a whole number of
            beats, so the energy smears across neighbouring bins. The
            autocorrelation finds the period directly. Then:

            * a peak below `acf_min` means the clip is **non_rhythmic**;
            * a period dividing the clip into a whole number of cycles (±0.1)
              means it is **rhythmic** after all;
            * otherwise the clip is **rhythmic_partial**, with
              `beats = frames / period`, moved into the BPM window by octaves.

            The plot is shown even when the spectrum was confident, for
            comparison.
        """),
        _code("""
            fig, ax = plt.subplots()
            for name, raw in (("motion", m), ("brightness", brightness)):
                if name not in signals:
                    continue
                ac = A.autocorrelation(A.normalise(raw))
                lag, peak = A.acf_period(ac, 4)
                half = len(ac) // 2
                ax.plot(np.arange(half + 1), ac[:half + 1], lw=1, label=name)
                if lag:
                    ax.axvline(lag, ls="--", color=ax.lines[-1].get_color())
                    print(f"{name}: period {lag:.2f} frames, peak {peak:.2f} -> {n / lag:.2f} cycles per clip")
                else:
                    print(f"{name}: no clear period")
            ax.axhline(PARAMS.acf_min, color="red", lw=0.8, label=f"acf_min {PARAMS.acf_min}")
            ax.set_xlabel("lag (frames)"); ax.set_title("autocorrelation", fontsize=9); ax.legend()
            plt.show()
            used = not (best and best.confidence >= PARAMS.min_confidence)
            print("\\nfallback used:", used)
        """),

        _md("""
            ## 7. Result

            The scanner's own `analyze()` on the same signals. This is exactly
            the row `beatscan` writes for this clip. It should agree with the
            step-by-step result above.
        """),
        _code("""
            a = A.analyze(motion, brightness, fps, PARAMS)
            row = {
                "status": a.status, "beats": a.beats, "bpm": None if a.bpm is None else round(a.bpm, 1),
                "confidence": round(a.confidence, 3), "alt_beats": a.alt_beats, "character": a.character,
                "signal": a.signal, "whole_loop": a.whole_loop, "loop_seamless": a.loop_seamless,
                "duration_s": round(duration, 3), "fps": round(fps, 3), "frames": n,
            }
            for key, value in row.items():
                print(f"{key:14s} {value}")
            if beats is not None and a.status == "rhythmic":
                assert a.beats == beats, "step-by-step and analyze() disagree"
        """),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"step-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="beatscan-explain",
                                description="Write a notebook that walks through the analysis of one clip.")
    p.add_argument("clip", type=Path)
    p.add_argument("-o", "--output", type=Path, help="notebook path (default: ./<clip name>_explained.ipynb)")
    p.add_argument("--no-execute", action="store_true", help="write the notebook without running it")
    p.add_argument("--min-confidence", type=float, default=Params.min_confidence)
    p.add_argument("--tolerance", type=float, default=Params.tolerance)
    p.add_argument("--bpm-range", default="80-180")
    args = p.parse_args(argv)
    if not args.clip.is_file():
        p.error(f"not a file: {args.clip}")
    try:
        lo, hi = (float(v) for v in args.bpm_range.split("-"))
    except ValueError:
        p.error("--bpm-range must look like 80-180")
    check_tools()

    clip = args.clip.resolve()
    probe(str(clip))  # fail early on files ffmpeg can't read
    params = Params(min_confidence=args.min_confidence, tolerance=args.tolerance, bpm_min=lo, bpm_max=hi)
    out = (args.output or Path(f"{clip.stem}_explained.ipynb")).resolve()
    out.write_text(json.dumps(build_notebook(clip, params), indent=1))

    if not args.no_execute:
        try:
            import nbclient  # noqa: F401
        except ImportError:
            print("nbclient/ipykernel not installed (pip install -e '.[notebook]'); "
                  "notebook written without outputs.", file=sys.stderr)
        else:
            print(f"Running {out.name} ...", file=sys.stderr)
            execute(out)
    print(f"Notebook: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
