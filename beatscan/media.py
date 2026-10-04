"""Probing and decoding through ffprobe/ffmpeg."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

from .analysis import SignalBuilder

WIDTH, HEIGHT = 64, 36

# Pixel formats that carry an alpha channel (pal8 may, via the palette).
_ALPHA_RE = re.compile(r"^(yuva|rgba|bgra|argb|abgr|gbrap|ya\d|ayuv|vuya|uyva|pal8)")


class MediaError(RuntimeError):
    pass


def check_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise SystemExit(
            f"beatscan: {' and '.join(missing)} not found on PATH. "
            "Install ffmpeg (e.g. `brew install ffmpeg`) and try again."
        )


@dataclass
class Probe:
    codec: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    nb_frames: int | None
    container_duration: float | None
    vfr: bool

    @property
    def has_alpha(self) -> bool:
        return bool(_ALPHA_RE.match(self.pix_fmt or ""))

    @property
    def est_duration(self) -> float | None:
        if self.nb_frames and self.fps:
            return self.nb_frames / self.fps
        return self.container_duration

    def to_dict(self) -> dict:
        return asdict(self)


def _rate(s: str | None) -> float:
    try:
        r = Fraction(s)
        return float(r) if r > 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _num(s, cast):
    try:
        return cast(s)
    except (TypeError, ValueError):
        return None


def probe(path: str) -> Probe:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration"
        ":format=duration",
        "-of", "json", path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise MediaError(f"ffprobe: {res.stderr.strip() or 'failed'}")
    info = json.loads(res.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise MediaError("no video stream")
    st = streams[0]
    r, avg = _rate(st.get("r_frame_rate")), _rate(st.get("avg_frame_rate"))
    fps = avg or r
    if not fps:
        raise MediaError("frame rate not reported")
    duration = _num(st.get("duration"), float) or _num(info.get("format", {}).get("duration"), float)
    return Probe(
        codec=st.get("codec_name", ""),
        pix_fmt=st.get("pix_fmt", ""),
        width=int(st.get("width") or 0),
        height=int(st.get("height") or 0),
        fps=fps,
        nb_frames=_num(st.get("nb_frames"), int) or None,
        container_duration=duration,
        vfr=bool(r and avg and abs(r - avg) / r > 0.01),
    )


def decode_frames(path: str, alpha: bool = False, width: int = WIDTH, height: int = HEIGHT,
                  stderr_out: list[str] | None = None):
    """Yield grey uint8 frames of shape (height, width), streamed from ffmpeg.

    Frames are read one at a time from the pipe, never buffered as a whole.
    Raises MediaError once the stream ends if ffmpeg failed. Any ffmpeg
    stderr text is appended to `stderr_out`.
    """
    vf = f"scale={width}:{height}:flags=area,format=gray"
    if alpha:
        # Composite over black: premultiplying by alpha is exactly that.
        vf = "format=rgba,premultiply=inplace=1," + vf
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path, "-an", "-sn", "-dn",
        "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-",
    ]
    frame_size = width * height
    count = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(frame_size)
                if len(buf) < frame_size:
                    break
                count += 1
                yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
        finally:
            proc.stdout.close()
            rc = proc.wait()
        err.seek(0)
        stderr = err.read().decode("utf-8", "replace").strip()
    if stderr and stderr_out is not None:
        stderr_out.append(stderr)
    if rc != 0:
        raise MediaError(f"ffmpeg exit {rc}: {stderr[-500:] or 'no output'}")
    if count == 0:
        raise MediaError(f"no frames decoded{': ' + stderr[-500:] if stderr else ''}")


def decode_signals(path: str, alpha: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    """Stream tiny grey frames from ffmpeg into motion/brightness signals.

    Returns (motion, brightness, stderr_text).
    """
    builder = SignalBuilder()
    stderr: list[str] = []
    for frame in decode_frames(path, alpha, stderr_out=stderr):
        builder.add(frame)
    motion, brightness = builder.finish()
    return motion, brightness, "\n".join(stderr)


def extract(path: str, max_duration: float) -> dict:
    """Worker entry point: probe + decode one clip. Never raises."""
    out: dict = {"warnings": []}
    try:
        pr = probe(path)
        out["probe"] = pr.to_dict()
        dur = pr.est_duration
        if dur is not None and dur > max_duration:
            out["skipped"] = f"longer than {max_duration:g}s ({dur:.1f}s)"
            return out
        if pr.vfr:
            out["warnings"].append("variable frame rate; treating frames as evenly spaced")
        motion, brightness, stderr = decode_signals(path, pr.has_alpha)
        if pr.nb_frames and pr.nb_frames != len(motion):
            out["warnings"].append(f"probe reported {pr.nb_frames} frames, decoded {len(motion)}")
        if stderr:
            out["warnings"].append(f"ffmpeg: {stderr.splitlines()[-1][:200]}")
        out["motion"] = motion.round(4).tolist()
        out["brightness"] = brightness.round(4).tolist()
    except Exception as e:  # one bad clip must never stop the scan
        out["error"] = str(e) or e.__class__.__name__
    return out
