"""Testing grounds runner: methods -> metrics -> before/after videos.

python run.py --clips hf_slowmo_girl --methods ffmpeg pp_current fusion
Outputs: DATA/clips/<clip>/out/<method>/, DATA/clips/<clip>/metrics.json,
         OUT/<clip>/sbs_<method>.mp4 (original | method), OUT/<clip>/grid.mp4, OUT/<clip>/metrics.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from common import OUT, clip_dir, has_method, list_clips, load_frames, load_meta, save_method, write_video
from flickerscope import analyze
from methods import METHODS
from metrics import score

ORDER = ["original", "ffmpeg", "pp_current", "aio_filter", "aio", "aio_hybrid_filter", "aio_hybrid", "flickerformer",
         "scope_gain", "fusion"]
LABEL = {"original": "Original", "ffmpeg": "ffmpeg deflicker", "pp_current": "Ours (previous)",
         "aio_filter": "AIO filter only", "aio": "AIO (CVPR'23)", "aio_hybrid_filter": "AIO filter + our guide",
         "aio_hybrid": "AIO-improved (our guide)", "flickerformer": "Flickerformer (CVPR'26)",
         "scope_gain": "FlickerScope gain", "fusion": "Flicker Fusion (new)"}


def label(img, text, sub=None):
    img = img.copy()
    s = max(0.5, img.shape[1] / 900)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    cv2.rectangle(img, (8, 8), (20 + tw, 22 + th + (int(26 * s) if sub else 0)), (0, 0, 0), -1)
    cv2.putText(img, text, (14, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (14, 14 + th + int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, s * 0.7, (255, 200, 120), 1,
                    cv2.LINE_AA)
    return img


def sub_for(res, m):
    r = res.get(m)
    return None if not r else f"flicker {r['flicker_pct']:.1f}%  Ewarp {r['ewarp']:.1f}"


def render(name, meta, orig, results, methods):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    avail = [m for m in ORDER if m == "original" or (m in methods and has_method(name, m))]
    clips = {m: (orig if m == "original" else load_frames(name, m)) for m in avail}
    T, H, W, _ = orig.shape
    for m in avail[1:]:
        sbs = np.empty((T, H, 2 * W, 3), np.uint8)
        pair = np.concatenate([orig, clips[m]], axis=2)  # clean original | method, for the viewer
        write_video(pair, os.path.join(d, f"pair_{m}.mp4"), meta["fps"], crf=16)
        for t in range(T):
            sbs[t, :, :W] = label(orig[t], "Original", sub_for(results, "original"))
            sbs[t, :, W:] = label(clips[m][t], LABEL.get(m, m), sub_for(results, m))
        write_video(sbs, os.path.join(d, f"sbs_{m}.mp4"), meta["fps"])
    n = len(avail)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    th, tw = H // 2 - (H // 2) % 2, W // 2 - (W // 2) % 2
    grid = np.zeros((T, rows * th, cols * tw, 3), np.uint8)
    for i, m in enumerate(avail):
        r, c = divmod(i, cols)
        for t in range(T):
            tile = cv2.resize(clips[m][t], (tw, th), interpolation=cv2.INTER_AREA)
            grid[t, r * th:(r + 1) * th, c * tw:(c + 1) * tw] = label(tile, LABEL.get(m, m), sub_for(results, m))
    write_video(grid, os.path.join(d, "grid.mp4"), meta["fps"])


def metric_version():
    """Hash of the meter's source: scores from a different FlickerScope/metrics version are recomputed,
    so every number in one table comes from the same meter."""
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for fn in ("flickerscope.py", "metrics.py", "common.py"):
        h.update(open(os.path.join(here, fn), "rb").read())
    return h.hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--methods", nargs="*", default=[m for m in ORDER if m != "original"])
    ap.add_argument("--force", nargs="*", default=[], help="methods to recompute")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    mver = metric_version()
    for name in a.clips or list_clips():
        meta = load_meta(name)
        orig = load_frames(name)
        mpath = os.path.join(clip_dir(name), "metrics.json")
        res = json.load(open(mpath)) if os.path.exists(mpath) else {}
        scope = analyze(orig, name, meta["fps"])
        res["original"] = score(name, meta, orig, orig, scope)
        res["original"]["mver"] = mver
        res["original"]["scope"] = {k: scope[k] for k in ["f0", "hz", "components", "periodic", "events", "strength"]}
        for m in a.methods:
            if m in METHODS and (m in a.force or not has_method(name, m)):
                t0 = time.time()
                fn = METHODS[m]
                out = fn(name, orig, meta, scope=scope) if m in ("scope_gain", "fusion") else fn(name, orig, meta)
                save_method(name, m, out)
                res.setdefault(m, {})["seconds"] = time.time() - t0
            if not has_method(name, m):
                continue
            if m in a.force or res.get(m, {}).get("mver") != mver:
                secs = res.get(m, {}).get("seconds")
                res[m] = score(name, meta, orig, load_frames(name, m), scope)
                res[m]["mver"] = mver
                if secs is not None:
                    res[m]["seconds"] = secs
        json.dump(res, open(mpath, "w"), indent=1)
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        json.dump(res, open(os.path.join(OUT, name, "metrics.json"), "w"), indent=1)
        print(f"\n== {name}  ({meta['note']})")
        print(f"   {'method':26s} {'flicker%':>8s} {'calm%':>6s} {'cover%':>7s} {'peak%':>6s} {'Ewarp':>6s} {'struct':>6s} "
              f"{'grain':>6s} {'expo':>5s} {'sec':>6s}")
        for m in ORDER:
            if m in res and "ewarp" in res[m]:
                r = res[m]
                print(f"   {LABEL.get(m, m):26s} {r['flicker_pct']:8.2f} {r.get('flicker_calm_pct', float('nan')):6.2f} {r['coverage_pct']:7.1f} {r['peak_pct']:6.1f} "
                      f"{r['ewarp']:6.2f} {r['structure']:6.3f} {r['noise_ratio']:6.2f} {r['exposure']:5.2f} "
                      f"{r.get('seconds', float('nan')):6.1f}")
        if not a.no_render:
            render(name, meta, orig, res, a.methods)


if __name__ == "__main__":
    main()
