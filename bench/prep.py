"""Build the benchmark clip set: frames + meta.json per clip, then the flow cache.

python prep.py            # build every clip that is missing, then flows
python prep.py --flow-only
python prep.py --only hf_slowmo_girl range

Two kinds of source:
  * dataset clips: frame folders of the All-In-One-Deflicker (Blind Video Deflickering) dataset, listed
    below by their path inside the dataset. The dataset root defaults to DATA/All-In-One-Deflicker-Dataset
    (where fetch_dataset.py puts it); override with "dataset_root" in clips.json or DEFLICKER_DATASET.
  * your own videos: listed in clips.json at the repository root (or the file named by DEFLICKER_CLIPS),
    see clips.example.json. Relative paths are resolved against the folder of that JSON file.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (DATA, DEV, FF, ROOT, clip_dir, flow_path, frame_paths, list_clips, load_frames,
                    raft_flow, to_tensor)

MAX_FRAMES = 200
MAX_EDGE = 1280

# name: (folder inside the dataset, fps, note)
DATASET_CLIPS = {f"hf_slowmo_{n}": (f"SlowMotion/Youtube2/{n}", 30.0, "HF slow-motion")
                 for n in ["bulb", "tv", "fan", "face", "wedding", "girl", "dog", "india", "miss", "rain", "smoke"]}
DATASET_CLIPS.update({f"hf_timelapse_{n}": (f"TimeLapse/Youtube/{n}", 25.0, "HF time-lapse")
                      for n in ["sunset", "city", "snownight", "night2day"]})


def clip_config():
    """Your own clips and the dataset root, from clips.json (missing file = dataset clips only).

    {"dataset_root": "data/All-In-One-Deflicker-Dataset",
     "videos": {"myclip": {"source": "videos/myclip.mov", "start": 0, "fps": 30.0, "note": "..."}}}
    Returns ({name: (source, start_frame, fps, note)}, dataset_root)."""
    path = os.environ.get("DEFLICKER_CLIPS", os.path.join(ROOT, "clips.json"))
    cfg = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: os.path.abspath(p if os.path.isabs(p) else os.path.join(base, p))
    videos = {}
    for name, v in cfg.get("videos", {}).items():
        src = rel(v["source"])
        fps = v.get("fps") or probe_fps(src)
        videos[name] = (src, int(v.get("start", 0)), float(fps), v.get("note", ""))
    root = os.environ.get("DEFLICKER_DATASET") or \
        (rel(cfg["dataset_root"]) if cfg.get("dataset_root") else os.path.join(DATA, "All-In-One-Deflicker-Dataset"))
    return videos, root


def probe_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def _fit(img):
    h, w = img.shape[:2]
    s = min(1.0, MAX_EDGE / max(h, w))
    nh, nw = int(round(h * s / 2) * 2), int(round(w * s / 2) * 2)
    return img if (nh, nw) == (h, w) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _finish(name, frames_bgr, fps, source, note):
    d = clip_dir(name)
    fd = os.path.join(d, "frames")
    if os.path.isdir(fd):
        shutil.rmtree(fd)
    os.makedirs(fd)
    for i, f in enumerate(frames_bgr):
        cv2.imwrite(os.path.join(fd, f"{i:05d}.png"), _fit(f), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    h, w = _fit(frames_bgr[0]).shape[:2]
    meta = dict(name=name, source=source, fps=fps, frames=len(frames_bgr), width=w, height=h, note=note)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {name}: {len(frames_bgr)} frames {w}x{h} @ {fps}fps")


def build_video(name, src, start, fps, note):
    tmp = os.path.join(DATA, "_tmp_extract")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    subprocess.run([FF, "-loglevel", "error", "-i", src, "-vf",
                    f"select='between(n\\,{start}\\,{start + MAX_FRAMES - 1})'", "-vsync", "0",
                    "-start_number", "0", os.path.join(tmp, "%05d.png")], check=True)
    frames = [cv2.imread(p) for p in frame_paths(tmp)]
    _finish(name, frames, fps, f"{src} [frames {start}..{start + len(frames) - 1}]", note)
    shutil.rmtree(tmp, ignore_errors=True)


def build_folder(name, folder, fps, note):
    paths = frame_paths(folder)[:MAX_FRAMES]
    if len(paths) < 20:
        print(f"  skip {name}: only {len(paths)} frames on disk")
        return
    _finish(name, [cv2.imread(p) for p in paths], fps, folder, note)


def illum_normalize(x):
    """Divide out low-frequency illumination so flicker does not bias optical flow."""
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    H, W = x.shape[-2:]
    small = F.avg_pool2d(lum, 8)
    blur = F.avg_pool2d(small, 9, stride=1, padding=4, count_include_pad=False)
    blur = F.interpolate(blur, size=(H, W), mode="bilinear", align_corners=False)
    return (x / (blur + 0.02) * 0.45).clamp(0, 1)


@torch.no_grad()
def build_flow(name, scale=0.5, frames=None, tag="", normalize=True, force=False, iters=20, max_side=1280):
    """RAFT flow cache. frames=None uses the clip; tag names an alternative cache (e.g. flow from a
    deflickered pass, where illumination normalisation is unnecessary). iters / max_side trade
    accuracy for speed (the benchmark uses the defaults; production may track faster)."""
    if os.path.exists(flow_path(name, tag)) and not force:
        return
    fr = load_frames(name) if frames is None else frames
    norm = illum_normalize if normalize else (lambda x: x)
    T, H, W, _ = fr.shape
    h, w = int(H * scale), int(W * scale)
    # RAFT's all-pairs correlation grows with the 4th power of frame size: 720x1280 needs < 1 GB, 1080x1920
    # over 4 GB per level, which overflows an 8 GB GPU into system RAM and crawls. Track at <= 1280 px on
    # the long side (the benchmark clips are all at or below it, so their flow is unchanged).
    cs = min(1.0, float(max_side) / max(H, W))
    Hc, Wc = int(round(H * cs)), int(round(W * cs))
    small = (lambda x: F.interpolate(x, size=(Hc, Wc), mode="area")) if cs < 1.0 else (lambda x: x)
    fw = np.zeros((T - 1, 2, h, w), np.float16)
    bw = np.zeros((T - 1, 2, h, w), np.float16)
    prev = small(norm(to_tensor(fr[0:1])))
    for t in range(T - 1):
        nxt = small(norm(to_tensor(fr[t + 1:t + 2])))
        fl = raft_flow(torch.cat([prev, nxt]), torch.cat([nxt, prev]), iters=iters)
        fl = F.interpolate(fl, size=(h, w), mode="bilinear", align_corners=False) * (w / Wc)
        fw[t] = fl[0].cpu().numpy()
        bw[t] = fl[1].cpu().numpy()
        prev = nxt
    np.savez(flow_path(name, tag), fw=fw, bw=bw, scale=scale)
    print(f"  flow{tag} {name}: {T - 1} pairs, mean |fw| {np.abs(fw).mean() / scale:.2f}px", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-only", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.flow_only:
        video_clips, dataset_root = clip_config()
        for n, (src, st, fps, note) in video_clips.items():
            if (not a.only or n in a.only) and not os.path.exists(os.path.join(clip_dir(n), "meta.json")):
                if not os.path.isfile(src):
                    print(f"  skip {n}: no video at {src}")
                    continue
                build_video(n, src, st, fps, note)
        for n, (sub, fps, note) in DATASET_CLIPS.items():
            folder = os.path.join(dataset_root, *sub.split("/"))
            if (not a.only or n in a.only) and not os.path.exists(os.path.join(clip_dir(n), "meta.json")) \
                    and os.path.isdir(folder):
                build_folder(n, folder, fps, note)
    for n in list_clips():
        if not a.only or n in a.only:
            build_flow(n)
