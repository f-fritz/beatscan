"""Pure analysis: per-frame signals in, beat count out.

Nothing in this module touches ffmpeg or the filesystem, so it can be tested
with plain numpy arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Harmonics summed per candidate (k, 2k, 3k, 4k). Hard cuts spread energy
# across harmonics, so the fundamental needs credit for them.
N_HARMONICS = 4
# Weight of energy that sits *between* the harmonics of a candidate (below
# N_HARMONICS * k). Penalises multiples of the true count (16 for an 8-pulse
# clip leaves energy at 8, 24, ...) without hurting the true count.
OFF_LATTICE_PENALTY = 0.5
# Raw signal levels are in 8-bit grey levels (0-255).
STATIC_MOTION = 0.5  # mean frame difference below which nothing visibly changes
FLAT_STD = 0.1  # a signal with less spread than this carries no information
MIN_FRAMES = 16
# Share of the motion signal's spike energy held by its largest frames above
# which the clip counts as "cuts" rather than "pulse".
CUTS_SHARE = 0.6
# When looking for the period in the autocorrelation, take the shortest lag
# whose peak is at least this fraction of the strongest one.
ACF_PEAK_RATIO = 0.85
# How many standard deviations above the noise level the lattice share must
# be before it counts towards confidence.
NULL_SIGMAS = 3.0
# How far (in factors of 2) to look for a beat count inside the BPM window.
OCTAVES = 4
# An autocorrelation peak must rise this much above the trough before it,
# so the slow decay of a trend is not mistaken for a period.
ACF_PROMINENCE = 0.2
# Fallback periods within this many cycles of a whole number count as a
# whole loop (rhythmic), not a partial one.
WHOLE_TOLERANCE = 0.1


@dataclass(frozen=True)
class Params:
    min_confidence: float = 0.5
    tolerance: float = 0.15
    bpm_min: float = 80.0
    bpm_max: float = 180.0
    acf_min: float = 0.6


class SignalBuilder:
    """Accumulate motion and brightness one frame at a time.

    The decoder streams frames into this, so the decoded video is never
    buffered; only two floats per frame are kept.
    """

    def __init__(self) -> None:
        self._first: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._motion: list[float] = []
        self._brightness: list[float] = []

    def add(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, dtype=np.float32)
        self._brightness.append(float(f.mean()))
        if self._prev is None:
            self._first = f
            self._motion.append(0.0)  # replaced by last->first in finish()
        else:
            self._motion.append(float(np.abs(f - self._prev).mean()))
        self._prev = f

    def __len__(self) -> int:
        return len(self._brightness)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        motion = np.array(self._motion, dtype=np.float64)
        if len(motion) > 1:
            # The clip is a loop: frame 0 follows the last frame.
            motion[0] = float(np.abs(self._first - self._prev).mean())
        return motion, np.array(self._brightness, dtype=np.float64)


def frame_signals(frames) -> tuple[np.ndarray, np.ndarray]:
    """Motion and brightness signals for an iterable of 2-D grey frames."""
    builder = SignalBuilder()
    for f in frames:
        builder.add(f)
    return builder.finish()


@dataclass
class SignalResult:
    name: str
    power: np.ndarray  # bin k = energy at k cycles per clip
    scores: np.ndarray  # harmonic score per k; NaN where k is not a candidate
    k: int
    confidence: float


@dataclass
class Candidate:
    beats: int
    score: float
    bpm: float


@dataclass
class Analysis:
    status: str
    beats: float | None = None
    confidence: float = 0.0
    alt_beats: list[int] = field(default_factory=list)
    character: str = ""
    signal: str = ""
    whole_loop: bool | None = None
    loop_seamless: bool | None = None
    bpm: float | None = None
    reason: str = ""
    # Debug information for plots.
    k_detected: int | None = None
    candidates: list[Candidate] = field(default_factory=list)
    signals: dict[str, SignalResult] = field(default_factory=dict)
    acf: np.ndarray | None = None
    acf_lag: float | None = None


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    sd = x.std()
    return x / sd if sd > 0 else x


def cycle_power(x: np.ndarray) -> np.ndarray:
    """Power spectrum over the full clip; bin k is k cycles per clip."""
    p = np.abs(np.fft.rfft(x)) ** 2
    p[0] = 0.0
    return p


def harmonic_scores(power: np.ndarray, kmax: int) -> np.ndarray:
    """Score each cycle count k in 2..kmax, relative to total energy.

    score = sum_h P[h*k] / h  -  OFF_LATTICE_PENALTY * (energy below N*k
    that is not on a multiple of k).
    """
    top = len(power) - 1
    scores = np.full(kmax + 1, np.nan)
    total = power[1:].sum()
    if total <= 0 or top < 2:
        return scores
    p = power / total
    hs = np.arange(1, N_HARMONICS + 1)
    for k in range(2, min(kmax, top) + 1):
        idx = hs * k
        ok = idx <= top
        on = float((p[idx[ok]] / hs[ok]).sum())
        upto = min(N_HARMONICS * k, top)
        off = float(p[1 : upto + 1].sum() - p[k : upto + 1 : k].sum())
        scores[k] = on - OFF_LATTICE_PENALTY * off
    return scores


def lattice_confidence(power: np.ndarray, k: int) -> float:
    """Share of energy on multiples of k, corrected for chance (0 = noise, 1 = exact).

    For noise, the share on the lattice averages c = (bins on lattice) / (bins)
    with spread sd ~ sqrt(c(1-c)/bins). Subtracting c + NULL_SIGMAS*sd keeps
    small k (where half the bins are on the lattice) from scoring by luck.
    """
    top = len(power) - 1
    total = power[1:].sum()
    if total <= 0 or k < 2 or k > top:
        return 0.0
    on = power[k::k].sum() / total
    c = (top // k) / top
    floor = c + NULL_SIGMAS * math.sqrt(c * (1 - c) / (top + 1))
    if floor >= 1:
        return 0.0
    return float(np.clip((on - floor) / (1 - floor), 0.0, 1.0))


def analyse_signal(name: str, x: np.ndarray) -> SignalResult:
    power = cycle_power(x)
    scores = harmonic_scores(power, max(2, len(x) // 4))
    if np.all(np.isnan(scores)):
        return SignalResult(name, power, scores, 0, 0.0)
    k = int(np.nanargmax(scores))
    return SignalResult(name, power, scores, k, lattice_confidence(power, k))


def pref_class(k: int) -> int:
    """0: power of two, 1: multiple of 4, 2: multiple of 3, 3: anything else."""
    if k >= 1 and k & (k - 1) == 0:
        return 0
    if k % 4 == 0:
        return 1
    if k % 3 == 0:
        return 2
    return 3


def resolve_beats(
    k0: int, scores: np.ndarray, duration: float, params: Params
) -> tuple[int, list[int], list[Candidate]]:
    """Turn a detected cycle count into a beat count plus alternatives.

    Candidates are cycle counts that score within `tolerance` of the best,
    plus the octave relatives of the best (k0 * 2^j), because the signal
    alone cannot tell 8 beats from 4 beats with a pulse every half beat. The
    BPM window and the power-of-two preference decide between them.
    """
    best = scores[k0] if k0 < len(scores) else np.nan
    scored = {k0}
    if not np.isnan(best):
        scored |= {
            k for k in range(2, len(scores))
            if not np.isnan(scores[k]) and scores[k] >= best - params.tolerance * abs(best)
        }
    octaves = {
        int(k0 * 2.0**j) for j in range(-OCTAVES, OCTAVES + 1)
        if k0 * 2.0**j >= 1 and (k0 * 2.0**j).is_integer()
    }

    def bpm(k: int) -> float:
        return k * 60.0 / duration

    def score(k: int) -> float:
        return float(scores[k]) if k < len(scores) and not np.isnan(scores[k]) else float("nan")

    inside = [k for k in scored | octaves if params.bpm_min <= bpm(k) <= params.bpm_max]
    pool = inside if inside else sorted(scored)

    def rank(k: int):
        s = score(k)
        return (pref_class(k), abs(math.log2(k / k0)), -(s if not math.isnan(s) else -1e9))

    pool.sort(key=rank)
    considered = [Candidate(k, score(k), bpm(k)) for k in sorted(scored | octaves)]
    return pool[0], pool[1:], considered


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised (unbiased) linear autocorrelation, lags 0..n-1."""
    n = len(x)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(np.abs(f) ** 2)[:n]
    if ac[0] <= 0:
        return np.zeros(n)
    ac = ac / ac[0] * n / (n - np.arange(n))
    return np.clip(ac, -1.0, 1.0)


def acf_period(ac: np.ndarray, min_lag: int) -> tuple[float | None, float]:
    """Shortest strong peak lag (sub-frame refined) and its height."""
    n = len(ac)
    max_lag = n // 2
    peaks = [
        i for i in range(max(min_lag, 1), min(max_lag, n - 2) + 1)
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]
        and ac[i] - ac[1:i].min() >= ACF_PROMINENCE
    ]
    if not peaks:
        return None, 0.0
    strongest = max(ac[i] for i in peaks)
    if strongest <= 0:
        return None, 0.0
    i = next(i for i in peaks if ac[i] >= ACF_PEAK_RATIO * strongest)
    a, b, c = ac[i - 1], ac[i], ac[i + 1]
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return i + float(np.clip(offset, -0.5, 0.5)), float(ac[i])


def character(motion: np.ndarray, k: int | None = None) -> str:
    """'cuts' when a few sharp spikes dominate the motion, 'pulse' when it varies smoothly."""
    d = np.clip(motion - np.median(motion), 0, None) ** 2
    if d.sum() <= 0:
        return "pulse"
    top = max(1, math.ceil(0.1 * len(d)), 2 * (k or 0))
    top = min(top, len(d) // 2) or 1
    share = np.sort(d)[-top:].sum() / d.sum()
    return "cuts" if share >= CUTS_SHARE else "pulse"


def loop_seamless(motion: np.ndarray) -> bool:
    """True if the last->first frame difference is no bigger than the jumps inside the clip."""
    if len(motion) < 3:
        return True
    return bool(motion[0] <= motion[1:].max() * 1.1 + 0.5)


def analyze(motion: np.ndarray, brightness: np.ndarray, fps: float, params: Params = Params()) -> Analysis:
    motion = np.asarray(motion, dtype=np.float64)
    brightness = np.asarray(brightness, dtype=np.float64)
    n = len(motion)
    if n < MIN_FRAMES:
        return Analysis("skipped", reason=f"only {n} frames")
    duration = n / fps

    seamless = loop_seamless(motion)
    m = motion.copy()
    if not seamless:
        # A seam jump is an artefact of the cut, not part of the rhythm.
        m[0] = np.median(m[1:])

    flat_m, flat_b = m.std() < FLAT_STD, brightness.std() < FLAT_STD
    if flat_m and flat_b:
        if motion[1:].mean() < STATIC_MOTION:
            return Analysis("static", loop_seamless=seamless, character="pulse")
        return Analysis("non_rhythmic", loop_seamless=seamless, character=character(m),
                        reason="constant motion")

    signals = {}
    if not flat_m:
        signals["motion"] = analyse_signal("motion", normalise(m))
    if not flat_b:
        signals["brightness"] = analyse_signal("brightness", normalise(brightness))
    best = max(signals.values(), key=lambda s: s.confidence)

    if best.confidence >= params.min_confidence:
        beats, alts, cands = resolve_beats(best.k, best.scores, duration, params)
        return Analysis(
            "rhythmic", beats=beats, confidence=best.confidence, alt_beats=alts,
            character=character(m, best.k), signal=best.name, whole_loop=True,
            loop_seamless=seamless, bpm=beats * 60.0 / duration,
            k_detected=best.k, candidates=cands, signals=signals,
        )

    # Fallback: periodic, but maybe not cut to a whole number of beats.
    min_lag = 4  # matches the n/4 upper limit on cycle counts
    acf_best = (None, 0.0, None, "")
    for name, raw in (("motion", m), ("brightness", brightness)):
        if name not in signals:
            continue
        ac = autocorrelation(normalise(raw))
        lag, peak = acf_period(ac, min_lag)
        if lag is not None and peak > acf_best[1]:
            acf_best = (lag, peak, ac, name)
    lag, peak, ac, name = acf_best

    if lag is not None and peak >= params.acf_min:
        cycles = n / lag
        k = round(cycles)
        if k >= 1 and abs(cycles - k) <= WHOLE_TOLERANCE:
            # Weak in the spectrum, but the period divides the clip evenly.
            beats, alts, cands = resolve_beats(k, signals[name].scores, duration, params)
            return Analysis(
                "rhythmic", beats=beats, confidence=peak, alt_beats=alts,
                character=character(m, k), signal=name, whole_loop=True,
                loop_seamless=seamless, bpm=beats * 60.0 / duration,
                k_detected=k, candidates=cands, signals=signals, acf=ac, acf_lag=lag,
            )
        beats = fit_bpm_range(cycles, duration, params)
        return Analysis(
            "rhythmic_partial", beats=round(beats, 2), confidence=peak,
            character=character(m), signal=name, whole_loop=False,
            loop_seamless=seamless, bpm=beats * 60.0 / duration,
            k_detected=best.k, signals=signals, acf=ac, acf_lag=lag,
        )

    return Analysis(
        "non_rhythmic", confidence=best.confidence, character=character(m),
        signal=best.name, loop_seamless=seamless, k_detected=best.k,
        signals=signals, acf=ac, acf_lag=lag,
    )


def fit_bpm_range(cycles: float, duration: float, params: Params) -> float:
    """Pick the octave of a fractional cycle count whose tempo falls in the BPM window."""
    for m in (1, 2, 0.5, 4, 0.25):
        bpm = cycles * m * 60.0 / duration
        if params.bpm_min <= bpm <= params.bpm_max:
            return cycles * m
    return cycles
