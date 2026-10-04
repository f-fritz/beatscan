"""Analysis tests on synthetic frames; no ffmpeg needed."""

import numpy as np
import pytest

from beatscan.analysis import Params, analyze, frame_signals, pref_class, resolve_beats
from synth import FPS, flashes, moving_shape, noise, sine_pulse, static, to_tiny


def run(frames, fps=FPS, params=Params()):
    motion, brightness = frame_signals(to_tiny(frames))
    return analyze(motion, brightness, fps, params)


@pytest.mark.parametrize("cycles", [4, 8, 16, 32])
def test_flashes(cycles):
    a = run(flashes(cycles))
    assert a.status == "rhythmic"
    assert a.beats == cycles
    assert a.whole_loop is True
    assert a.character == "cuts"
    assert a.confidence > 0.9


def test_sine_pulse():
    a = run(sine_pulse(8))
    assert (a.status, a.beats, a.character) == ("rhythmic", 8, "pulse")


def test_moving_shape_without_brightness_change():
    a = run(moving_shape(4))
    assert (a.status, a.beats, a.signal) == ("rhythmic", 4, "motion")


@pytest.mark.parametrize("cycles", [3, 12])
def test_non_power_of_two(cycles):
    a = run(flashes(cycles))
    assert (a.status, a.beats) == ("rhythmic", cycles)


def test_static():
    assert run(static()).status == "static"


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [60, 120, 240])
def test_noise(seed, n):
    assert run(noise(n, seed=seed)).status == "non_rhythmic"


def test_partial():
    a = run(flashes(7.5, period=16))
    assert a.status == "rhythmic_partial"
    assert a.whole_loop is False
    assert a.beats == pytest.approx(7.5, abs=0.1)


def test_seam_jump_detected():
    frames = sine_pulse(8)
    frames[-1] = 0  # hard jump into frame 0
    a = run(frames)
    assert a.loop_seamless is False
    assert a.beats == 8


def test_bpm_window_doubles_slow_pulse():
    # 8 flashes over 8 s is 60 BPM; 16 beats (120 BPM) is the plausible reading.
    a = run(flashes(8, period=30))
    assert (a.status, a.beats) == ("rhythmic", 16)
    assert a.bpm == pytest.approx(120)


def test_bpm_window_can_be_moved():
    a = run(flashes(8, period=30), params=Params(bpm_min=50, bpm_max=70))
    assert a.beats == 8


def test_too_short():
    assert run(flashes(1, period=10)).status == "skipped"


def test_resolve_prefers_power_of_two():
    scores = np.full(40, -1.0)
    scores[12] = scores[16] = 0.5
    # 12 and 16 tie; both in range for an 8 s clip (90 and 120 BPM).
    beats, alts, _ = resolve_beats(12, scores, 8.0, Params())
    assert beats == 16
    assert 12 in alts


def test_pref_class():
    assert [pref_class(k) for k in (1, 8, 12, 6, 5)] == [0, 0, 1, 2, 3]
