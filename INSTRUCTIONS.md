# Beats-per-clip scanner for VJ content

## Goal

Build a command-line tool that scans a directory of VJ video clips and, for each clip, determines:

1. Whether the clip is rhythmically periodic (i.e. usable with beat-synced playback).
2. If so, **how many beats the clip contains** (e.g. 4, 8, 16, 32).

The beat count is the deliverable. BPM is not an output of interest: these clips are loops that get stretched to whatever tempo is playing, so what matters is the number of beats the loop spans. BPM may only be used internally as a plausibility check (see "Resolving ambiguity").

## Non-goals

- No modification, renaming, or re-encoding of source clips.
- No audio analysis. Ignore audio tracks entirely.
- No writing of metadata into VJ software. Output is a report file only.

## Stack

- Python 3.10+
- `ffmpeg` and `ffprobe` on PATH for all decoding (do not use OpenCV for decoding; codec support for HAP/DXV/ProRes is better through ffmpeg).
- `numpy` for the analysis. `matplotlib` only for the optional debug plots.
- Check at startup that ffmpeg and ffprobe are available and exit with a clear message if not.

## CLI

```
beatscan <directory> [options]

  -o, --output PATH        Report path (default: ./beatscan.csv)
  --ext LIST               Extensions to scan (default: mov,mp4,avi,mkv,webm,m4v)
  --workers N              Parallel workers (default: CPU count - 1)
  --min-confidence FLOAT   Threshold for the "rhythmic" flag (default: tune in testing)
  --bpm-range MIN-MAX      Plausibility window for disambiguation (default: 80-180)
  --max-duration SEC       Skip clips longer than this (default: 120)
  --no-cache               Ignore the cache and re-analyse everything
  --plots DIR              Write one debug plot per clip into DIR
```

Scan recursively. Show a progress bar or a running count.

## Analysis pipeline (per clip)

### 1. Probe

Use `ffprobe` to get frame rate, frame count, and duration. Derive duration as `frame_count / fps` rather than trusting the container duration. If the frame count is not reported, count the frames actually decoded.

### 2. Decode to a tiny grayscale stream

Pipe raw frames from ffmpeg at very low resolution, at the native frame rate (no frame rate conversion):

```
ffmpeg -v error -i <clip> -an -vf "scale=64:36,format=gray" -f rawvideo -
```

Clips with an alpha channel must be composited over black before the grayscale conversion, otherwise transparent regions produce garbage.

### 3. Build per-frame signals

Compute two signals of length `n_frames`:

- **Motion**: mean absolute difference between each frame and the previous one. Treat the clip as a loop, so frame 0 is compared against the last frame.
- **Brightness**: mean luminance per frame.

Remove the mean from each and normalise to unit variance. If the variance of both is effectively zero, classify the clip as `static` and stop.

### 4. Find cycles per clip

Because the clip is a loop, the useful question is "how many times does the pattern repeat across the whole clip?". Take the FFT of each signal over the full clip length. Bin `k` then corresponds directly to `k` cycles per clip, which is the candidate beat count. No tempo estimate is needed.

For each signal:

- Compute the magnitude spectrum, ignoring bin 0.
- For each candidate `k` (1 up to roughly `n_frames / 4`), compute a score combining the energy at `k` with the energy at its harmonics `2k`, `3k`, `4k`. Hard cuts and flashes spread energy across harmonics, so harmonic summing matters.
- Express the score relative to the total spectral energy so it is comparable between clips.

Combine the motion and brightness results by taking the stronger one, and record which signal won.

### 5. Resolving ambiguity

The raw result is ambiguous by factors of 2 (a clip with 8 pulses could be 8 beats, or 4 beats with a pulse on every half beat, or 16 beats with a pulse every two beats). Resolve it in this order:

1. Collect all candidates whose score is within a tolerance of the best score.
2. Compute the implied tempo for each: `bpm = k * 60 / duration`. Discard candidates outside `--bpm-range` if at least one candidate is inside it.
3. Among the remainder, prefer powers of two (4, 8, 16, 32, 64), then multiples of 4, then multiples of 3 (6, 12, 24 for triplet or 3/4 material).
4. Report the winner as `beats` and the other surviving candidates as `alt_beats`.

Never report BPM as a headline result. `implied_bpm` may appear in the report only as a sanity-check column.

### 6. Fallback for non-loop clips

If no bin stands out (energy is smeared across neighbouring bins), the clip may be rhythmic but not cut to a whole number of beats. In that case:

- Compute the normalised autocorrelation of the motion signal and find the strongest peak beyond a minimum lag.
- Report `beats = duration_frames / peak_lag` as a fractional value and set `whole_loop = false`.

### 7. Classification

Assign each clip one status:

- `rhythmic`: confidence above threshold, whole number of beats.
- `rhythmic_partial`: periodic but not a whole number of beats (fallback path).
- `non_rhythmic`: has motion but no clear periodicity.
- `static`: effectively no change between frames.
- `skipped`: too long, or too short (fewer than ~16 frames).
- `error`: decode failure. Record the ffmpeg error text.

Also record a `character` value: `cuts` when the motion signal is dominated by a few sharp spikes, `pulse` when it varies smoothly.

## Output

A CSV with one row per clip:

| Column | Meaning |
|---|---|
| `path` | Path relative to the scanned directory |
| `status` | One of the classifications above |
| `beats` | Detected beats in the clip (integer, or fractional for partial) |
| `confidence` | 0 to 1 |
| `alt_beats` | Other plausible beat counts, semicolon separated |
| `character` | `cuts` or `pulse` |
| `signal` | `motion` or `brightness`, whichever gave the result |
| `whole_loop` | true/false |
| `loop_seamless` | true if the last-to-first frame difference is in line with typical in-clip differences |
| `duration_s`, `fps`, `frames` | Clip properties |
| `implied_bpm` | Sanity check only |
| `error` | Error text if any |

Sort the CSV by status, then by confidence descending. Print a short summary to the terminal at the end (counts per status, counts per beat value).

## Robustness and performance

- One failing clip must never stop the scan. Catch errors per clip and record them.
- Process clips in parallel with a process pool. Each worker runs its own ffmpeg subprocess.
- Cache results in a JSON file next to the report, keyed by path, file size, and modification time, so re-runs only analyse new or changed clips.
- Memory per clip is tiny at 64x36, but still stream frames from the pipe rather than buffering the whole decode output for very long clips.
- Handle variable frame rate files by logging a warning and treating frames as evenly spaced.

## Testing

Create a test module that generates synthetic clips with ffmpeg or numpy and asserts the detected beat count:

- White flash every N frames, with 4, 8, 16, and 32 flashes per clip.
- Smooth sine-wave brightness pulse with 8 cycles per clip.
- A moving shape that completes 4 cycles per clip (motion without brightness change).
- A 3-cycle and a 12-cycle clip (non power-of-two).
- A static colour clip (expect `static`).
- Random noise (expect `non_rhythmic`).
- A flash clip trimmed to 7.5 cycles (expect `rhythmic_partial`).

Use the synthetic set to tune the default `--min-confidence` and the candidate tolerance in step 5, and document the chosen values in the README.

## Debug plots

When `--plots` is given, save one PNG per clip showing the motion signal over time, the cycles-per-clip spectrum with the chosen `k` marked, and the candidates considered. This is the main tool for checking wrong results by eye.

## Deliverables

- `beatscan/` package with separate modules for probing/decoding, analysis, and CLI/reporting. Keep the analysis functions pure (numpy arrays in, results out) so they can be tested without ffmpeg.
- Tests as described above.
- A short README covering install, usage, the column meanings, and known limitations (half/double ambiguity, clips with no visual rhythm, clips that are not cut to whole beats).

## Open points to confirm with the user

- Which codecs the library uses (HAP, DXV, ProRes, H.264). Verify the installed ffmpeg build decodes them and report any that fail.
- Whether very long clips (non-loop footage) should be analysed in windows rather than skipped.
- Whether a later step should write the beat counts into the VJ software's clip settings. Out of scope for now.