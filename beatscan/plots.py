"""Optional debug plots (needs matplotlib)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import Analysis


def write_plot(out: Path, title: str, a: Analysis, motion, brightness, fps: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion = np.asarray(motion)
    brightness = np.asarray(brightness)
    t = np.arange(len(motion)) / fps
    xmax = max(40, (a.beats or 0) * 2.5)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"{title}\n{a.status}  beats={a.beats}  conf={a.confidence:.2f}  "
                 f"signal={a.signal or '-'}  alt={a.alt_beats or '-'}", fontsize=10)

    ax = axes[0]
    ax.plot(t, motion, lw=1, label="motion")
    ax.set_ylabel("motion (grey levels)")
    ax2 = ax.twinx()
    ax2.plot(t, brightness, lw=1, color="tab:orange", alpha=0.7, label="brightness")
    ax2.set_ylabel("brightness")
    if a.beats and a.whole_loop is not None:
        for b in np.arange(0, len(motion) / fps, len(motion) / fps / a.beats):
            ax.axvline(b, color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel("seconds")
    ax.set_title("signals (grey lines = reported beat grid)", fontsize=9)

    ax = axes[1]
    for name, s in a.signals.items():
        k = np.arange(len(s.power))
        rel = s.power / s.power[1:].sum() if s.power[1:].sum() > 0 else s.power
        ax.plot(k[1:], rel[1:], lw=1, label=f"{name} power")
    if a.k_detected:
        ax.axvline(a.k_detected, color="red", lw=1, ls="--", label=f"detected k={a.k_detected}")
    if a.beats and a.whole_loop:
        ax.axvline(a.beats, color="green", lw=1.5, label=f"reported beats={a.beats}")
    ax.set_xlabel("cycles per clip (k)")
    ax.set_ylabel("share of energy")
    ax.set_xlim(0, xmax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.set_title("cycles-per-clip spectrum", fontsize=9)

    ax = axes[2]
    if a.signals:
        for name, s in a.signals.items():
            ks = np.arange(len(s.scores))
            ax.plot(ks, s.scores, marker=".", lw=1, label=f"{name} score")
        for c in (c for c in a.candidates if c.beats <= xmax):
            colour = "green" if c.beats == a.beats else "orange"
            ax.axvline(c.beats, color=colour, lw=1, alpha=0.6)
            ax.annotate(f"{c.beats}\n{c.bpm:.0f}bpm", (c.beats, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="top")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("cycles per clip (k)")
        ax.set_ylabel("harmonic score")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("candidate scores (vertical lines = candidates considered)", fontsize=9)
    if a.acf is not None:
        inset = ax.inset_axes([0.6, 0.55, 0.38, 0.4])
        inset.plot(a.acf[: len(a.acf) // 2 + 1], lw=1)
        if a.acf_lag:
            inset.axvline(a.acf_lag, color="red", lw=1)
        inset.set_title("autocorrelation", fontsize=7)
        inset.tick_params(labelsize=6)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
