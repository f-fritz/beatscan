"""End-to-end tests: encode synthetic clips with ffmpeg, then scan them."""

import csv
import json
import shutil
import subprocess

import numpy as np
import pytest

from beatscan import cli
from beatscan.analysis import analyze
from beatscan.media import extract, probe
from synth import FPS, flashes, moving_shape, noise, sine_pulse, static

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def encode(path, frames, pix_fmt="gray", codec=("-c:v", "ffv1")):
    """Write frames (n,h,w) grey or (n,h,w,4) RGBA to a lossless clip."""
    n, h, w = frames.shape[:3]
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", *codec, str(path),
    ]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    return path


def scan(path):
    data = extract(str(path), 120)
    assert "error" not in data, data.get("error")
    return analyze(np.array(data["motion"]), np.array(data["brightness"]), data["probe"]["fps"])


@pytest.mark.parametrize("cycles", [4, 8, 16, 32])
def test_flash_clip(tmp_path, cycles):
    a = scan(encode(tmp_path / "f.mkv", flashes(cycles)))
    assert (a.status, a.beats) == ("rhythmic", cycles)


def test_shape_clip(tmp_path):
    a = scan(encode(tmp_path / "s.mkv", moving_shape(4)))
    assert (a.status, a.beats) == ("rhythmic", 4)


def test_alpha_is_composited_over_black(tmp_path):
    # Colour channels are noise everywhere; only alpha carries the flashes.
    fl = flashes(8)
    rgba = np.random.default_rng(0).integers(0, 256, fl.shape + (4,), dtype=np.uint8)
    rgba[..., 3] = fl
    path = encode(tmp_path / "a.mov", rgba, "rgba", ("-c:v", "qtrle", "-pix_fmt", "argb"))
    assert probe(str(path)).has_alpha
    a = scan(path)
    assert (a.status, a.beats) == ("rhythmic", 8)


def test_frame_count_matches(tmp_path):
    path = encode(tmp_path / "c.mkv", sine_pulse(8))
    data = extract(str(path), 120)
    assert len(data["motion"]) == 120
    assert data["probe"]["fps"] == pytest.approx(FPS)


def test_broken_file_reports_error(tmp_path):
    bad = tmp_path / "bad.mov"
    bad.write_bytes(b"not a video")
    assert "error" in extract(str(bad), 120)


def test_cli_end_to_end(tmp_path, capsys):
    clips = tmp_path / "clips"
    (clips / "sub").mkdir(parents=True)
    encode(clips / "flash16.mkv", flashes(16))
    encode(clips / "sub" / "sine8.mkv", sine_pulse(8))
    encode(clips / "static.mkv", static())
    encode(clips / "noise.mkv", noise(120))
    encode(clips / "partial.mkv", flashes(7.5, period=16))
    (clips / "broken.mov").write_bytes(b"junk")
    (clips / "notes.txt").write_text("ignored")

    report = tmp_path / "out.csv"
    args = [str(clips), "-o", str(report), "--workers", "2", "--plots", str(tmp_path / "plots")]
    assert cli.main(args) == 0
    rows = {r["path"]: r for r in csv.DictReader(open(report))}

    assert set(rows) == {"flash16.mkv", "sub/sine8.mkv", "static.mkv", "noise.mkv",
                         "partial.mkv", "broken.mov"}
    assert (rows["flash16.mkv"]["status"], rows["flash16.mkv"]["beats"]) == ("rhythmic", "16")
    assert (rows["sub/sine8.mkv"]["status"], rows["sub/sine8.mkv"]["beats"]) == ("rhythmic", "8")
    assert rows["static.mkv"]["status"] == "static"
    assert rows["noise.mkv"]["status"] == "non_rhythmic"
    assert rows["partial.mkv"]["status"] == "rhythmic_partial"
    assert rows["partial.mkv"]["whole_loop"] == "false"
    assert rows["broken.mov"]["status"] == "error" and rows["broken.mov"]["error"]
    assert len(list((tmp_path / "plots").glob("*.png"))) == 5

    statuses = [r["status"] for r in csv.DictReader(open(report))]
    assert statuses == sorted(statuses, key=["rhythmic", "rhythmic_partial", "non_rhythmic",
                                             "static", "skipped", "error"].index)

    # Second run is served from the cache (5 decodable clips).
    cache = tmp_path / "out.cache.json"
    assert cache.exists()
    assert len(json.loads(cache.read_text())["entries"]) == 5
    assert cli.main(args[:-2]) == 0
    assert "5 cached, analysing 1" in capsys.readouterr().err


def test_max_duration_skips(tmp_path):
    clips = tmp_path / "c"
    clips.mkdir()
    encode(clips / "long.mkv", flashes(16))  # 8 s
    report = tmp_path / "r.csv"
    assert cli.main([str(clips), "-o", str(report), "--max-duration", "5", "--workers", "1"]) == 0
    row = next(csv.DictReader(open(report)))
    assert row["status"] == "skipped"
