"""Synthetic benchmark clips with ground truth.

Clean base: one still frame (no temporal flicker by construction) animated with a smooth camera
move plus a moving occluder patch, so motion handling is exercised. It is relit with a random
physically modelled flicker configuration (synth.py). Saved as a normal clip (frames + meta.json)
plus gt/ (clean frames) and gt.npz (gain field, per-pixel Percent Flicker, video %).

python prep_synth.py --n 8 --seed 1

Stills are drawn from the clips in SRC_CLIPS that you have prepared (prep.py), or from any prepared
non-synthetic clip if none of those exist. The published synthetic set used the author's clip set, so
other source clips give different (equally valid) synthetic clips.
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np

from common import DATA, clip_dir, frame_paths, list_clips
from prep import build_flow
from synth import apply, sample_config, video_pct

# the clips the published synthetic set drew its stills from (range, elec, house: private test clips,
# not distributed; the rest are from the Blind Video Deflickering dataset, see prep.py)
SRC_CLIPS = ["range", "elec", "house", "hf_slowmo_face", "hf_slowmo_wedding", "hf_slowmo_tv", "hf_timelapse_city",
             "hf_slowmo_girl", "hf_slowmo_dog", "hf_slowmo_india"]


def clean_base(rng, T=120, out_hw=(360, 640)):
    names = [n for n in SRC_CLIPS if os.path.isdir(os.path.join(clip_dir(n), "frames"))]
    if not names:
        names = [n for n in list_clips() if not n.startswith(("synth_", "_"))]
    if not names:
        raise SystemExit("no source clips: run prep.py first (synthetic clips relight stills from real clips)")
    name = str(rng.choice(names))
    paths = frame_paths(os.path.join(clip_dir(name), "frames"))
    still = cv2.cvtColor(cv2.imread(str(rng.choice(paths))), cv2.COLOR_BGR2RGB)
    occl = cv2.cvtColor(cv2.imread(str(rng.choice(paths))), cv2.COLOR_BGR2RGB)
    oh, ow = out_hw
    if still.shape[0] > still.shape[1]:  # keep landscape output: rotate portrait stills
        still = np.ascontiguousarray(np.rot90(still))
    big = cv2.resize(still, (int(ow * 1.35), int(oh * 1.35)), interpolation=cv2.INTER_AREA)
    # smooth camera path (pan + slight zoom)
    p0, p1 = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
    z0, z1 = rng.uniform(1.0, 1.15), rng.uniform(1.0, 1.15)
    ph, pw = int(oh * 0.35), int(ow * 0.22)  # occluder patch
    patch = cv2.resize(occl[occl.shape[0] // 3:occl.shape[0] // 3 + occl.shape[0] // 3,
                            occl.shape[1] // 3:occl.shape[1] // 3 + occl.shape[1] // 3], (pw, ph))
    a0, a1 = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
    frames = []
    for n in range(T):
        s = 0.5 - 0.5 * np.cos(np.pi * n / (T - 1))  # ease in/out
        z = z0 + (z1 - z0) * s
        cw, ch = ow * z, oh * z
        cx = (big.shape[1] - cw) * (p0[0] + (p1[0] - p0[0]) * s) + cw / 2
        cy = (big.shape[0] - ch) * (p0[1] + (p1[1] - p0[1]) * s) + ch / 2
        M = np.float32([[1 / z, 0, ow / 2 - cx / z], [0, 1 / z, oh / 2 - cy / z]])
        f = cv2.warpAffine(big, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ax = int((ow - pw) * (a0[0] + (a1[0] - a0[0]) * s)); ay = int((oh - ph) * (a0[1] + (a1[1] - a0[1]) * s))
        f[ay:ay + ph, ax:ax + pw] = patch
        frames.append(f)
    return np.stack(frames), name


def build(i, rng):
    name = f"synth_{i:02d}"
    d = clip_dir(name)
    shutil.rmtree(d, ignore_errors=True)
    clean, src = clean_base(rng)
    T, H, W, _ = clean.shape
    cfg = sample_config(rng, H, W)
    flick, G, pct = apply(clean, cfg, rng, t_offset=float(rng.uniform(0, 1)))
    for sub, arr in [("frames", flick), ("gt", clean)]:
        os.makedirs(os.path.join(d, sub))
        for n, f in enumerate(arr):
            cv2.imwrite(os.path.join(d, sub, f"{n:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    gt_pct = video_pct(pct, clean)
    np.savez_compressed(os.path.join(d, "gt.npz"), G=G.astype(np.float16), pct=pct, video_pct=gt_pct)
    desc = ", ".join(f"{s['kind']}" + (f"@{s['params']['pwm_hz']:.0f}Hz d{s['params']['duty']:.2f}" if s["kind"] == "led_pwm" else "")
                     for s in cfg["sources"])
    note = (f"SYNTHETIC GT: {desc}; {cfg['f_mains']:.0f}Hz mains, {cfg['fps']:.0f}fps, shutter "
            f"{cfg['exposure'] * cfg['fps']:.2f}, {'rolling' if cfg['line_time'] else 'global'}; GT flicker {gt_pct:.1f}%")
    cfgj = {k: v for k, v in cfg.items() if k != "sources"}
    cfgj["sources"] = [{"kind": s["kind"], "params": s["params"], "weight": s["weight"]} for s in cfg["sources"]]
    meta = dict(name=name, source=f"still from {src}", fps=30.0, frames=T, width=W, height=H, note=note,
                synthetic=True, gt_flicker_pct=gt_pct, config=cfgj)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    build_flow(name)
    print(f"{name}: {note}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for i in range(a.start, a.start + a.n):
        build(i, rng)
