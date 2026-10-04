"""Synthetic test clips as numpy frame stacks (uint8, grey or RGBA).

Default period is 15 frames at 30 fps, i.e. one pulse per beat at 120 BPM,
so every clip's true beat count sits inside the default BPM window.
"""

from __future__ import annotations

import numpy as np

FPS = 30
PERIOD = 15
H, W = 72, 128


def flashes(cycles: float, period: int = PERIOD, h: int = H, w: int = W) -> np.ndarray:
    n = int(round(cycles * period))
    frames = np.zeros((n, h, w), np.uint8)
    frames[::period] = 255
    return frames


def sine_pulse(cycles: int, period: int = PERIOD, h: int = H, w: int = W) -> np.ndarray:
    n = cycles * period
    level = 128 + 100 * np.sin(2 * np.pi * np.arange(n) / period)
    return np.broadcast_to(level[:, None, None], (n, h, w)).round().astype(np.uint8).copy()


def moving_shape(cycles: int, period: int = PERIOD, h: int = H, w: int = W) -> np.ndarray:
    """A disc orbiting at varying speed; mean brightness stays constant."""
    n = cycles * period
    yy, xx = np.mgrid[0:h, 0:w]
    frames = np.zeros((n, h, w), np.uint8)
    r = h * 0.12
    for t in range(n):
        phase = 2 * np.pi * t / period
        angle = phase + 0.6 * np.sin(phase)  # speed changes once per cycle
        cx = w / 2 + w * 0.3 * np.cos(angle)
        cy = h / 2 + h * 0.3 * np.sin(angle)
        frames[t][(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 255
    return frames


def static(n: int = 120, h: int = H, w: int = W) -> np.ndarray:
    return np.full((n, h, w), 90, np.uint8)


def noise(n: int = 120, h: int = H, w: int = W, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, h, w), dtype=np.uint8)


def to_tiny(frames: np.ndarray, th: int = 36, tw: int = 64) -> np.ndarray:
    """Area-average downscale, standing in for ffmpeg's scale=64:36."""
    n, h, w = frames.shape
    return frames.reshape(n, th, h // th, tw, w // tw).mean(axis=(2, 4))
