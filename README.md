# Deflicker Fusion — FlickerScope + Flicker Fusion

**Video deflicker and flicker removal for slow-motion footage shot under LED and fluorescent lights.**
A physically grounded flicker *meter* (FlickerScope) and a flicker *remover* (Flicker Fusion): per pixel,
motion-compensated, no training. A video deflickering method that removes flicker from slo-mo video
(120–960 fps), follows camera and subject motion with optical flow, and keeps real flashes.

[Before/after demo (Hugging Face Space)](https://huggingface.co/spaces/VerySilent/deflicker-fusion) ·
[Method write-up](METHOD.md) ·
[Deflicker for iPhone](https://deflickerapp.com) ([App Store](https://apps.apple.com/us/app/deflicker-slow-mo-flicker-fix/id6790765901)) ·
[License: PolyForm Noncommercial 1.0.0](LICENSE)

## What it is

Slow-motion (slo-mo) video at 120–960 fps samples mains-powered lights faster than they stay steady.
LED drivers, fluorescent ballasts and even filament bulbs pulse at twice the mains frequency
(100 or 120 Hz), so the frames alternate between bright and dark. That flicker is multiplicative, lives
in linear light, is aliased by the frame rate (100 Hz at 240 fps repeats every 12 frames), differs for
every light in the room, gets row-shifted by a rolling shutter, and has to be told apart from real events
such as a camera flash or a muzzle flash. Generic deflickering (one global gain per frame, temporal
smoothing, neural atlases) models none of this.

- **FlickerScope** (the meter) fits, for every pixel along optical-flow trajectories, a model
  `Y = S · G` of steady scene light `S` times a flicker gain `G`. It finds the lamp frequency and its
  aliased harmonics, separates one-off events from flicker, and reports IEEE 1789 **Percent Flicker** and
  **Flicker Index** per pixel and for the whole video. On synthetic clips with known ground truth its
  video score correlates with the truth at r = 0.988.
- **Flicker Fusion** (the remover) rebuilds every frame from its flow-aligned neighbours, each divided by
  its own measured gain and weighted by signal-to-noise, so a frame caught at the lamps' darkest point is
  rebuilt mostly from bright, clean neighbours instead of being brightened (and its noise amplified)
  14×. Clipped highlights, flashes and moving subjects get their own rules, and two guards make sure it
  never adds flicker locally and never shifts the clip's exposure.
- **No training**: the method has no learned weights of its own. Its only network is a pretrained
  optical-flow model (RAFT), fetched separately.
- Output keeps the original resolution (rounded down to even dimensions for H.264), frame timing and
  audio.

## Headline results

All methods are scored by the same meter (FlickerScope at the original's flicker frequency, frozen
settings). *Calm flicker* leaves out the frames around real events, so no method gains from smoothing
away a flash. *Warp error* (×1000) is the field-standard temporal-consistency metric and does not use
our model. *Structure* is the SSIM structure term against the original (1.0 = picture untouched).
Lower is better except for structure.

**Shooting range** (200 frames of 240 fps slow motion under 50 Hz LED lighting, muzzle flash at frame 157;
the author's own clip):

| method | flicker % | calm flicker % | warp error | structure | grain | time |
|---|---|---|---|---|---|---|
| original | 29.73 | 29.19 | 58.69 | 1.000 | 0.90 | — |
| ffmpeg `deflicker` | 33.89 | 33.27 | 63.01 | 0.999 | 0.94 | 11 s |
| previous method (temporal median) | 8.83 | 8.20 | 18.36 | 0.989 | 0.95 | 45 s |
| All-In-One-Deflicker (CVPR'23) | 2.28 | 1.59 | 7.53 | 0.754 | 0.75 | ~65 min |
| AIO filter guided by our estimate | 2.48 | 1.66 | 8.82 | 0.734 | 0.87 | ~5 min |
| **Flicker Fusion** | **1.98** | **1.24** | **5.21** | **0.986** | **0.68** | ~3 min |

ffmpeg makes this clip worse: one global gain cannot fix flicker that only part of the frame has. The
CVPR 2023 method erases the muzzle flash's fireball; Flicker Fusion keeps it. Times on one RTX 4070
(8 GB); Flicker Fusion's time excludes the RAFT optical-flow pass it shares with the meter (a few
minutes per 200 frames).

**Real-world slow-motion clips** from the
[Blind Video Deflickering dataset](https://huggingface.co/datasets/xrenaa/Blind-Video-Deflickering-Dataset)
(same method, same settings, never tuned on these):

| clip | flicker % before → after | warp error before → after | structure | AIO (CVPR'23): flicker / warp / structure |
|---|---|---|---|---|
| night walkway (`girl`) | 48.4 → **3.2** | 50.6 → 6.0 | 0.992 | not run |
| living room, TV light (`tv`) | 11.2 → **1.6** | 8.1 → 5.3 | 0.996 | 3.6 / 3.8 / 0.646 |
| dog in a hallway (`dog`) | 12.4 → **1.6** | 16.0 → 2.0 | 0.995 | not run |
| stage lights (`india`) | 4.5 → **1.1** | 14.0 → 6.2 | 0.997 | not run |
| close-up, night lights (`face`) | 53.2 → **19.0** | 71.9 → 9.6 | 0.981 | 19.6 / 7.6 / 0.828 |
| flame-effect bulb (`bulb`) | 3.4 → **1.5** | 9.4 → 7.1 | 0.998 | 1.4 / 4.7 / 0.699 |
| falling water (`rain`), partial | 67.3 → 12.3 | 77.5 → 22.0 | 0.994 | not run |
| ceiling fan, fluorescent tube (`fan`), partial | 11.1 → 3.5 | 31.1 → 14.4 | 0.997 | 2.7 / 6.1 / 0.894 |
| ballroom first dance (`wedding`), partial | 33.9 → 28.3 | 11.9 → 8.6 | 0.975 | 26.5 / 6.7 / 0.662 |
| outdoor portrait (`miss`), partial | 25.9 → 20.0 | 6.5 → 4.3 | 0.991 | not run |

Where the CVPR 2023 method also ran it sometimes reads lower flicker (`fan`, `wedding`, `bulb`) and
has the lower warp error on all five, but it rewrites much more of the picture (structure 0.65–0.89
against 0.975–0.998 for Flicker Fusion). "Partial" clips improve but keep visible flicker; see
[Limitations](#limitations).

**Synthetic clips with ground truth** (8 clips: still frames animated with camera motion and a moving
occluder, relit by simulated filament, fluorescent, LED-driver and PWM-dimmed LED sources on 50/60 Hz
mains, 120–480 fps, global or rolling shutter, clipping and sensor noise):

| clip | true flicker % | FlickerScope reads | PSNR original | PSNR best other | PSNR Flicker Fusion |
|---|---|---|---|---|---|
| LED + PWM dimmer, 120 fps | 1.2 | 1.4 | 34.9 | 34.9 | **41.4** |
| 2 kHz PWM, 480 fps | 3.8 | 3.7 | 33.0 | 33.0 | **35.8** |
| 3 sources incl. PWM | 11.2 | 11.5 | 30.1 | 30.6 | **35.7** |
| fluorescent, rolling shutter | 12.5 | 14.6 | 30.5 | 32.0 | **41.4** |
| 2 tubes, 480 fps | 12.8 | 16.5 | 28.0 | 29.5 | **37.1** |
| 3 sources, 240 fps | 19.0 | 20.1 | 28.2 | 32.5 | **38.3** |
| 3 lamps out of phase | 23.3 | 30.6 | 27.3 | 29.2 | **31.6** |
| raw LED driver | 38.8 | 41.5 | 23.5 | 29.3 | **35.5** |

Meter: correlation with the truth r = 0.988, mean absolute error 2.2 points, flicker frequency correct
on 8 of 8, median per-pixel map correlation 0.80. Remover: Flicker Fusion wins every clip by 2.4 to
9.5 dB PSNR over the best other result (best of the untouched original, ffmpeg `deflicker`, the previous
method and a FlickerScope-gain-only ablation).

**Flickerformer** (Qu et al., CVPR 2026): `bench/flickerformer_baseline.py` runs it with its published
inference; its scores are not published here yet.

Numbers come from `out/<clip>/metrics.json` and `out/flickerscope_validation.json` (17 Oct 2026); within
each clip every method is scored with the same meter version.

## Before / after

Drag-to-compare players for the shooting-range clip (Flicker Fusion, the CVPR 2023 method, ffmpeg,
the previous method and the live flicker map) and for the dataset clips above:
**<https://huggingface.co/spaces/VerySilent/deflicker-fusion>**

## Quick start

Needs Python 3.10+ and an NVIDIA GPU with CUDA (developed on Python 3.12, PyTorch 2.4, RTX 4070 8 GB;
CPU-only execution is untested).

```bash
git clone https://github.com/verysilentco/deflicker-fusion
cd deflicker-fusion
pip install -r requirements.txt        # install a CUDA build of PyTorch first if pip would pick a CPU one

# RAFT optical flow (code + raft-things.pth) comes from an All-In-One-Deflicker checkout (not vendored)
mkdir -p third_party && cd third_party
git clone https://github.com/ChenyangLEI/All-In-One-Deflicker
cd All-In-One-Deflicker
git clone https://github.com/ChenyangLEI/cvpr2023_deflicker_public_folder
mv cvpr2023_deflicker_public_folder/pretrained_weights ./ && rm -rf cvpr2023_deflicker_public_folder
cd ../..
```

Deflicker a video:

```bash
python bench/deflicker_file.py path/to/clip.mov
```

This writes, next to the input (or into `--out`):

- `clip_deflickered.mp4`: the deflickered video, H.264, original frame timing, original audio copied;
- `clip_compare.mp4`: original and deflickered side by side;
- `clip_report.json`: flicker % before and after, detected frequency, events, per chunk.

| argument | default | meaning |
|---|---|---|
| `video` | (required) | input video |
| `--out DIR` | input's folder | where the three outputs go |
| `--chunk N` | 200 | frames per chunk; long videos run in overlapping chunks so GPU memory stays bounded |
| `--overlap N` | 24 | frames shared by neighbouring chunks (crossfaded when stitching) |
| `--max-frames N` | all | process only the first N frames (quick test) |
| `--fresh` | off | ignore saved progress and start over (an interrupted run otherwise resumes after the last finished chunk) |
| `--keep` | off | keep the work folder afterwards |
| `--adaptive` | off | motion-adaptive Fusion, for fast-moving subjects and faces |

The setup block above is bash (Git Bash works on Windows); the `python ...` commands are the same in
PowerShell. Prefer a local web page? `python app/server.py`
starts the **Deflicker Lab** on <http://127.0.0.1:8770/>: drop a video, pick the method, and compare
before and after in a wipe player. It runs on your own GPU and binds to 127.0.0.1 only.

Where things go (all optional; defaults are relative to the repository):

| variable | default | what |
|---|---|---|
| `DEFLICKER_DATA` | `./data` | working data: clips, frames, flow caches, method outputs, job folders |
| `DEFLICKER_OUT` | `./out` | rendered videos, maps, metrics, viewer manifest |
| `DEFLICKER_RAFT` | `./third_party/All-In-One-Deflicker` | checkout with the RAFT code and `pretrained_weights/raft-things.pth` |
| `DEFLICKER_DATASET` | `$DEFLICKER_DATA/All-In-One-Deflicker-Dataset` | benchmark dataset root |
| `DEFLICKER_CLIPS` | `./clips.json` | your own benchmark clips (see `clips.example.json`) |
| `DEFLICKER_FLICKERFORMER` | `./third_party/Flickerformer` | Flickerformer checkout (baseline only) |

## How it works

FlickerScope linearises the frames, computes RAFT optical flow on illumination-normalised frames (so the
flicker does not bias the motion), and finds the flicker fundamental from spectra of the frame-to-frame
change of log luminance *along the flow*, so camera and subject motion cancel. The fundamental's
harmonics are folded into the frame rate's band, and only those a short window can resolve are kept.
Per pixel and frame, a robust fit (Huber, then Tukey) over flow-aligned neighbours gives the periodic
gain; the spatially smooth remainder is aperiodic flicker; large one-off deviations are events (a flash),
and whole frames dominated by an event keep only the lamp model. The gain is normalised to its
arithmetic mean, so correcting deep flicker does not darken the image.

Flicker Fusion upsamples the gain edge-aware to full resolution, warps every neighbour within a few
frames onto the current one, divides each by its own gain and merges them with weights for
signal-to-noise, clipping (a highlight is rebuilt only if the clipping is explained by flicker), two-band
robustness (tolerant on brightness, strict on structure) and events (never borrowed). Flash light is
treated as added light: flash frames are rebuilt from their calm neighbours, with the lamps' light read
from the same lamp phase one light cycle earlier, and the flash added back on top.
A second, full-resolution pass refines the gain for thin and sharply bounded objects. Finally a
do-no-harm guard blends back to the original wherever the output fluctuates more than the input did, and
an exposure guard returns the original untouched if the model failed.

Full write-up, with the failures that shaped each rule: **[METHOD.md](METHOD.md)**.

## Reproduce the benchmark

```bash
python bench/fetch_dataset.py              # dataset frames (CC BY 4.0) -> data/All-In-One-Deflicker-Dataset/
cp clips.example.json clips.json           # optional: add your own clips
python bench/prep.py                       # frames + meta.json + RAFT flow per clip -> data/clips/<clip>/
python bench/prep_synth.py --n 8 --seed 7  # synthetic clips with ground truth
python bench/validate_scope.py             # FlickerScope vs ground truth -> out/flickerscope_validation.json
python bench/run.py                        # every method, scores, before/after renders -> out/<clip>/
python bench/aio_baseline.py hf_slowmo_tv  # optional: All-In-One-Deflicker baseline (~1 h per 200 frames)
python bench/flickerformer_baseline.py     # optional: Flickerformer baseline (needs its checkout + weights)
python bench/run.py --no-render            # scores any new baseline outputs
python bench/scope_viz.py hf_slowmo_girl   # FlickerScope figures and flicker-overlay video for a clip
python bench/manifest.py                   # viewer manifest
python -m http.server 8765                 # then open http://localhost:8765/viewer/
```

The testing-grounds viewer plays every clip and method in sync (wipe, side by side, grid, flicker map)
with the metrics table. Keys: `Space` play, `,` `.` frame step, `1`–`9` method, `←` `→` move the wipe,
hold `A`/`B` to peek, `W` `S` `G` `F` views, Ctrl+wheel zoom, `?` help.

Method changes are general rules checked on every clip, not per-clip patches:
`python bench/gate.py fusion_adaptive` runs a candidate against `fusion` on every prepared clip and fails
it if any clip gets worse on flicker, calm flicker, warp error, structure, grain, exposure or an artifact
score.

The results page (the Hugging Face Space) lives in `site/`: `bench/make_page.py` renders `index.html`
from `site/template.html` and the JSON next to it (CPU only), and `bench/build_site.py --examples` /
`--limits` renders the example videos, posters and tables from your benchmark outputs into `site/`.
The range clip's brightness curve (`curves.json`) and flicker-map images were exported separately.

The shooting-range clip and two other private test clips are not distributed, and the synthetic set
was built from stills of those clips too, so your synthetic clips (and the gate's clip set) will differ
from the published ones.

## Limitations

- **Built for lamp flicker in high-frame-rate footage.** Time-lapse exposure flicker is not periodic
  lamp flicker, and too much changes between shots to borrow light: on the dataset's time-lapse clips the
  reading barely moves (sunset 25.9 → 24.5, city 25.1 → 25.2, snow day 28.9 → 25.6, snow night
  40.3 → 39.6), though warp error drops on all of them.
- **Content optical flow cannot track.** Drifting smoke changes brightness with no lamp cycle
  (63.0 → 59.4); falling water improves only partly (brightness swing 2.8× → 1.8×, and the meter
  under-reads what is left because it books the deepest dips as events).
- **Periodic motion** (fan blades repeating every few frames) can read as flicker.
- **Brightness, not colour**: a tube whose colour cycles white to yellow still cycles after correction.
- **Very slow swings** (about 1 Hz) are only partly removed.
- The meter **over-reads slightly** (+0.3 to +7 points) with several out-of-phase sources and a
  near-full shutter.
- When the model fails, the guards return the original (locally or for the whole clip), so "no change"
  is a possible result.
- Needs an NVIDIA GPU. On an RTX 4070, Flicker Fusion takes about 3 minutes per 200 frames of
  720×1280, plus a few minutes of optical flow. Flow is computed at up to 1280 px on the long side.
- Research code: interfaces may change.

## Use it on iPhone

**[Deflicker: Slow Mo Flicker Fix](https://apps.apple.com/us/app/deflicker-slow-mo-flicker-fix/id6790765901)** on the App Store fixes
flicker in your iPhone slow-motion videos. Its Fusion engine runs this method. Guides, prices and
[how we test](https://deflickerapp.com/method/) are on [deflickerapp.com](https://deflickerapp.com).

## FAQ

**Why does my slow-motion video flicker?** Lights on mains power pulse at 100 Hz (50 Hz countries) or
120 Hz (60 Hz countries). At 240 fps each frame catches a different point of that pulse, so frames
alternate between bright and dark. Normal-speed video averages the pulse away; slow motion does not.

**Does it remove LED flicker and fluorescent flicker?** Yes, that is what it is built for: rectified LED
drivers, PWM-dimmed LEDs (tested from 100 Hz to 2 kHz), fluorescent tubes and filament bulbs, including
several lights with different phases in one scene and rolling-shutter banding.

**How is it different from ffmpeg's `deflicker` filter?** ffmpeg applies one brightness gain per frame.
Real scenes flicker differently in different places (above, ffmpeg makes the shooting-range clip
worse); Deflicker Fusion measures and corrects every pixel.

## Repository layout

| path | what |
|---|---|
| `bench/deflicker_file.py` | the command-line tool: any video in, deflickered video + compare video + report out |
| `bench/flickerscope.py` | **FlickerScope**: gain field, Percent Flicker maps, events, video score |
| `bench/methods.py` | **Flicker Fusion** and the baselines it is compared with (ffmpeg, previous method, gain-only ablation) |
| `bench/common.py` | clip I/O, colour (sRGB ↔ linear), RAFT optical flow, warping, occlusion-checked flow chains |
| `bench/metrics.py` | scores: residual and calm flicker, warp error, structure, grain, exposure, PSNR vs ground truth |
| `bench/synth.py`, `bench/prep_synth.py` | physically modelled synthetic flicker with ground truth |
| `bench/prep.py`, `bench/fetch_dataset.py` | benchmark clips (frames + meta + flow cache), dataset download |
| `bench/run.py`, `bench/manifest.py` | benchmark runner, before/after renders, viewer manifest |
| `bench/gate.py` | regression gate across every clip |
| `bench/validate_scope.py`, `bench/scope_viz.py`, `bench/instability.py` | meter validation, FlickerScope figures, residual-instability maps |
| `bench/aio_baseline.py`, `bench/flickerformer_baseline.py` | All-In-One-Deflicker (CVPR'23) and Flickerformer (CVPR'26) baselines |
| `bench/build_site.py`, `bench/make_page.py`, `site/` | the results page published as the Hugging Face Space |
| `viewer/index.html` | testing-grounds viewer (synced wipe / side by side / grid / flicker map + metrics) |
| `app/` | Deflicker Lab, a local web page for running your own clips (standard library only) |
| `METHOD.md` | the method write-up |

## Citation

If you use FlickerScope or Flicker Fusion in your work, please cite:

```bibtex
@misc{silentco2026deflickerfusion,
  title        = {Deflicker Fusion: FlickerScope and Flicker Fusion for Video Deflickering},
  author       = {{Silent Co, LLC}},
  year         = {2026},
  howpublished = {\url{https://github.com/verysilentco/deflicker-fusion}},
  url          = {https://github.com/verysilentco/deflicker-fusion}
}
```

The benchmark also relies on RAFT (Teed and Deng, ECCV 2020) and the Blind Video Deflickering dataset
(Lei et al., CVPR 2023); please cite them as well when you use the benchmark. See
[THIRD_PARTY.md](THIRD_PARTY.md).

## License

Copyright 2026 Silent Co, LLC. Licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**:
free to use, modify and share for any noncommercial purpose, including personal projects, research,
education and use by charitable, educational, public research and government organisations.
**Commercial use needs a separate license: contact@silent.co**

See [NOTICE](NOTICE) for the required notice and [THIRD_PARTY.md](THIRD_PARTY.md) for the components
and data you fetch separately (RAFT, All-In-One-Deflicker, Flickerformer, the benchmark dataset), each
under its own license.
