# beatscan

Scans a folder of VJ loops and reports **how many beats each loop spans** (4, 8, 16, 32, …), judged from its visual rhythm. Audio is ignored and source files are never modified.

## Install

Requires Python 3.10+ and `ffmpeg`/`ffprobe` on `PATH` (`brew install ffmpeg`).

```sh
pip install -e .              # numpy only
pip install -e '.[plots,test]'  # + matplotlib for --plots, pytest
```

## Usage

```sh
beatscan <directory> [options]       # or: python -m beatscan <directory>

  -o, --output PATH        Report path (default: ./beatscan.csv)
  --ext LIST               Extensions to scan (default: mov,mp4,avi,mkv,webm,m4v)
  --workers N              Parallel workers (default: CPU count - 1)
  --min-confidence FLOAT   Threshold for the "rhythmic" flag (default: 0.5)
  --tolerance FLOAT        Score tolerance for alternative beat counts (default: 0.15)
  --bpm-range MIN-MAX      Plausibility window for disambiguation (default: 80-180)
  --max-duration SEC       Skip clips longer than this (default: 120)
  --no-cache               Ignore the cache and re-analyse everything
  --plots DIR              Write one debug plot per clip into DIR
```

The directory is scanned recursively. Decoded signals are cached in `<report>.cache.json` next to the report, keyed by path, size and mtime. Re-runs only decode new or changed clips, and changing analysis options does not force a re-decode.

When a result looks wrong, run with `--plots DIR` and open that clip's PNG. It shows the motion and brightness signals with the reported beat grid, the cycles-per-clip spectrum, and the candidate scores with their implied BPM.

## Report columns

| Column | Meaning |
|---|---|
| `path` | Path relative to the scanned directory |
| `status` | `rhythmic`, `rhythmic_partial`, `non_rhythmic`, `static`, `skipped`, `error` |
| `beats` | Beats in the clip. Integer for `rhythmic`, fractional for `rhythmic_partial` |
| `confidence` | 0–1. Spectral periodicity for whole loops, autocorrelation peak for the fallback |
| `alt_beats` | Other plausible beat counts inside the BPM window, `;`-separated |
| `character` | `cuts` (a few sharp spikes dominate the motion) or `pulse` (smooth variation) |
| `signal` | `motion` or `brightness`, whichever gave the result |
| `whole_loop` | `true` if the clip spans a whole number of beats |
| `loop_seamless` | `true` if the last→first frame jump is no bigger than jumps inside the clip |
| `duration_s`, `fps`, `frames` | Clip properties; duration = decoded frames / fps |
| `implied_bpm` | Sanity check only, not a result |
| `codec` | Video codec reported by ffprobe |
| `error` | ffmpeg error text, or the reason a clip was skipped |

Rows are sorted by status, then confidence (highest first). A summary of counts per status, beat count and codec is printed at the end.

## How it works

1. Decode at native frame rate to 64×36 grey. Clips with alpha are composited over black first.
2. Build two per-frame signals: **motion** (mean absolute difference to the previous frame, with frame 0 compared to the last frame) and **brightness** (mean level).
3. FFT each signal over the whole clip, so bin *k* means *k* cycles per clip. Score each *k* by the energy at *k*, 2*k*, 3*k* and 4*k* (weighted 1, ½, ⅓, ¼), minus a penalty for energy between those harmonics. The penalty stops 16 from beating 8 on an 8-pulse clip. Confidence is the share of energy on multiples of *k*, measured relative to what noise would give. The stronger signal wins.
4. Resolve the factor-of-2 ambiguity. Candidates are counts scoring within `--tolerance` of the best, plus the best count ×2ⁿ. Candidates whose implied BPM falls outside `--bpm-range` are dropped, unless none fall inside. The rest are ranked: powers of two first, then multiples of 4, then multiples of 3, then whichever is closest to the detected count.
5. If confidence is below the threshold, fall back to the autocorrelation of each signal. A strong peak whose period divides the clip into a whole number of cycles (within 0.1) is reported as `rhythmic`. Otherwise it is reported as `rhythmic_partial` with a fractional beat count. Below 0.6 the clip is `non_rhythmic`.

## Tuned defaults

Chosen with the synthetic set in `tests/synth.py`, then checked against a set of real clips:

- **`--min-confidence 0.5`**: random-noise clips peak at 0.30 (99th percentile 0.05 over 200 runs). A flash train cut to 7.5 cycles scores 0.34, so it falls through to the partial path. Clean synthetic loops score 1.0 and real rhythmic clips 0.6–1.0.
- **`--tolerance 0.15`**: wide enough for genuine near-ties (e.g. 12 vs 16) to show up in `alt_beats`, narrow enough that unrelated counts don't.
- **Autocorrelation threshold 0.6** (`Params.acf_min`): noise reaches 0.48 at the 99th percentile and 0.68 at worst. Real partial loops sit around 0.65–1.0.

The other constants live at the top of `beatscan/analysis.py`.

## Tests

```sh
pytest
```

`tests/test_analysis.py` runs the analysis on synthetic frames with no ffmpeg needed. It covers flashes at 4/8/16/32 cycles, an 8-cycle sine pulse, a moving shape at 4 cycles, 3 and 12 cycles, a static clip, noise, 7.5 cycles, a seam jump and BPM disambiguation. `tests/test_pipeline.py` encodes the same kinds of clips with ffmpeg, including an alpha clip, and runs the full CLI with cache and plots.

## Known limitations

- **Half/double ambiguity is musical, not visual.** A clip pulsing 8 times could be 8, 16 or 4 beats. The BPM window picks one, so a clip actually made at 60 or 240 BPM is reported at the octave that fits 80–180. Check `implied_bpm` and `alt_beats`, or narrow `--bpm-range` for a set of clips you know.
- **Back-and-forth motion counts double.** Frame differences have no direction, so a pendulum produces two motion pulses per swing. The BPM window usually absorbs this as an octave.
- **No visual rhythm, no answer.** One-shot animations (a single rise and fall), slow morphs and very irregular bursts come out `non_rhythmic`, even if the creator cut them to 16 beats. Lowering `--min-confidence` admits more of them, at the cost of false positives.
- **Slow swells dominate.** When a clip's main movement is a 2-cycle swell, the beat count comes from that swell scaled into the BPM window, not from faster detail on top of it.
- **Loops not cut to whole beats** (`rhythmic_partial`) get a fractional beat count from the autocorrelation, which is less precise than the whole-loop path.
- **Variable frame rate** files trigger a warning and are treated as evenly spaced.
