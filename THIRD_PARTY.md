# Third-party components, data and credits

Nothing listed here is included in this repository. Each item is fetched separately by you, under its
own license. Flicker Fusion and FlickerScope themselves use no All-In-One-Deflicker or Flickerformer
code or weights; they only need an optical-flow network (RAFT).

## RAFT optical flow (required)

- **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**, Zachary Teed and Jia Deng,
  ECCV 2020. Princeton Vision & Learning Lab, <https://github.com/princeton-vl/RAFT>.
  License: **BSD-3-Clause**.
- Used for every flow field in the meter, the remover and the warping-error metric
  (`bench/common.py: raft()`, `raft_flow()`).
- Loaded at runtime from a checkout of **All-In-One-Deflicker** (next section), which carries the RAFT
  code under `src/models/stage_1/core/` and the `raft-things.pth` weights in its `pretrained_weights/`.

## All-In-One-Deflicker (required for RAFT; optional as a baseline)

- **Blind Video Deflickering by Neural Filtering with a Flawed Atlas**, Chenyang Lei, Xuanchi Ren,
  Zhaoxiang Zhang and Qifeng Chen, CVPR 2023. <https://github.com/ChenyangLEI/All-In-One-Deflicker>.
  License: **Apache-2.0**.
- Provides the RAFT code and weights above, and is the "AIO (CVPR'23)" baseline
  (`bench/aio_baseline.py`, which runs their published pipeline inside the checkout).

Setup (follow their README; the commands below are what it describes):

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/ChenyangLEI/All-In-One-Deflicker
cd All-In-One-Deflicker
git clone https://github.com/ChenyangLEI/cvpr2023_deflicker_public_folder
mv cvpr2023_deflicker_public_folder/pretrained_weights ./ && rm -rf cvpr2023_deflicker_public_folder
# needed by this repo: pretrained_weights/raft-things.pth
# needed only for the AIO baseline: pretrained_weights/neural_filter.pth, local_refinement_net.pth
```

The default location is `third_party/All-In-One-Deflicker`; set `DEFLICKER_RAFT` to use a checkout
elsewhere. Running the AIO baseline itself needs that project's own dependencies (its
`environment.yml`) in the same Python environment.

## Flickerformer (optional baseline)

- **It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal**, Lishen Qu,
  Shihao Zhou, Jie Liang, Hui Zeng, Lei Zhang and Jufeng Yang, CVPR 2026.
  <https://github.com/qulishen/Flickerformer>, project page
  <https://qulishen.github.io/project/Flickerformer.html>.
- `bench/flickerformer_baseline.py` re-implements their published inference procedure (three-frame
  input at 512x512, residual added back at full resolution) and loads, from **your own checkout**, their
  architecture file `basicsr/archs/Flickerformer_arch.py` and their weights `Flickerformer.pth`. Their
  code is built on BasicSR; the script replaces only BasicSR's registry decorator with a stub.
- When this benchmark was set up (their commit `56f2bef`, 31 Mar 2026) the Flickerformer repository
  had **no license file**. Check the authors' current terms before using it beyond your own evaluation.
- Default location `third_party/Flickerformer`; override with `DEFLICKER_FLICKERFORMER`.

## Benchmark clips: Blind Video Deflickering dataset

- The real-world slow-motion and time-lapse clips (`hf_slowmo_*`, `hf_timelapse_*`) come from the
  dataset released with All-In-One-Deflicker (Lei et al., CVPR 2023):
  <https://huggingface.co/datasets/xrenaa/Blind-Video-Deflickering-Dataset> (file
  `All-In-One-Deflicker-Dataset.zip`). License on the dataset card: **CC BY 4.0**.
- Not redistributed here. `bench/fetch_dataset.py` downloads only the folders the benchmark uses.
- The results page (Hugging Face Space) shows short before/after excerpts of some of these clips,
  side by side with our processed output, credited to the dataset.

## Other tools

- **FFmpeg**, used through the `imageio-ffmpeg` package (frame extraction, H.264 encoding, and its
  `deflicker` filter as the "ffmpeg" baseline). Installed by pip under FFmpeg's own license.
- **PyTorch, NumPy, OpenCV, SciPy, Matplotlib, remotezip**: installed separately (`requirements.txt`),
  each under its own license.
- **IBM Plex** fonts (SIL Open Font License 1.1), loaded from Google Fonts by the Lab and the
  results page.
