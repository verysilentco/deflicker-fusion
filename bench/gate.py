"""Regression gate: a change to the method ships only if no clip gets worse.

For each clip: run the candidate method (unless --no-run), score candidate and baseline with the same
meter and the same analysis of the original, and compare. A clip FAILS if the candidate is worse than
the baseline beyond a small tolerance on any of
  flicker %, calm-segment flicker %, warp error, structure, grain, exposure, artifact %
or, on the shooting-range clip, if the muzzle flash or the beam after it changes.

artifact % = share of visible, unclipped pixels (flash frames excluded) whose fine structure differs
from the original frame relit by the method's own smooth light, beyond the local noise: ghosts,
doubled edges, hard-edged patches. Light changes do not count, noise removal barely does.

python gate.py fusion_strict                      -> every benchmark clip in DATA, writes OUT/gate_<cand>.json
python gate.py fusion_strict --clips range hf_slowmo_face --no-run
Run one clip per process on long sweeps: a single process accumulates GPU memory.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import OUT, has_method, list_clips, load_frames, load_meta, save_method, srgb_to_lin, to_tensor
from flickerscope import analyze
from methods import METHODS
from metrics import score


def default_clips():
    """Every prepared benchmark clip (dataset, your own and synthetic); work folders of deflicker_file.py
    (names starting with '_') are not benchmark clips."""
    return [c for c in list_clips() if not c.startswith("_")]


# ---------------------------------------------------------------- image helpers for artifact_pct
def box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), 2 * r + 1, stride=1)


def wbox(x, w, r):
    return box(x * w, r) / box(w, r).clamp_min(1e-6)


def guided_w(I, p, w, r, eps):
    """Weighted guided filter: smooths p along the edges of I, ignoring pixels with w = 0."""
    mI, mp = wbox(I, w, r), wbox(p, w, r)
    a = (wbox(I * p, w, r) - mI * mp) / (wbox(I * I, w, r) - mI * mI + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def luma(x):
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def flash_frames(processed, jump=1.25, pad=3, half=6):
    """Frames lit by a flash (a muzzle flash, a guest's camera): after deflickering, the frame's mean
    linear brightness jumps above its +-half-frame neighbourhood (range shot: 1.29x and 1.99x; a
    camera flash at a wedding: 1.36x; residual flicker and fast motion stay below 1.21x).
    Returned padded by +-pad frames."""
    small = np.stack([cv2.resize(f, (160, 90) if f.shape[1] >= f.shape[0] else (90, 160), interpolation=cv2.INTER_AREA)
                      for f in processed]).astype(np.float32) / 255.0
    m = srgb_to_lin(small).mean((1, 2, 3))
    T = len(m)
    med = np.array([np.median(m[max(0, t - half):t + half + 1]) for t in range(T)])
    hit = m > jump * med
    out = hit.copy()
    for d in range(1, pad + 1):
        out[d:] |= hit[:-d]
        out[:-d] |= hit[d:]
    return out


# (key, better, absolute tolerance, relative tolerance)
CHECKS = [("flicker_pct", "lower", 0.3, 0.05), ("flicker_calm_pct", "lower", 0.3, 0.05), ("ewarp", "lower", 0.2, 0.05),
          ("structure", "higher", 0.005, 0.0), ("noise_ratio", "lower", 0.05, 0.0), ("artifact_pct", "lower", 0.1, 0.10)]


@torch.no_grad()
def artifact_pct(original, processed, k_lo=3.0, k_hi=6.0, clip_thr=0.97, chunk=8, step=2):
    T, H, W, _ = original.shape
    r = max(4, int(round(W / 48)))
    skip = flash_frames(processed)
    num = den = 0.0
    for t0 in range(0, T, chunk * step):
        ts = [t for t in range(t0, min(T, t0 + chunk * step), step) if not skip[t]]
        if not ts:
            continue
        Is = to_tensor(original[ts])
        Il, Ol = srgb_to_lin(Is), srgb_to_lin(to_tensor(processed[ts]))
        gI, gO = torch.log(luma(Il) + 1e-3), torch.log(luma(Ol) + 1e-3)
        clipped = (Is.amax(1, keepdim=True) >= clip_thr).float()
        Es = guided_w(gI, gO - gI, 1 - clipped, r, 0.04)
        hp = lambda g: g - box(g, 2)
        d = box((hp(gO) - hp(gI + Es)).abs(), 2)
        loc = torch.maximum(box(d, max(8, W // 16)), d.flatten(1).median(1).values[:, None, None, None]).clamp_min(1e-3)
        w = ((d - k_lo * loc) / ((k_hi - k_lo) * loc)).clamp(0, 1)
        vis = (1 - clipped) * (luma(Il) > 0.002).float()
        num += float((w * vis).sum())
        den += float(vis.sum())
    return 100.0 * num / max(den, 1.0)


def shot_checks(processed):
    """The range clip's muzzle flash core (frame 157) and the beam after it (frame 164): mean pixel value of
    fixed crops, which must not change (the flash is a real event, not flicker)."""
    return dict(flash_core=float(processed[157][470:560, 330:420].mean()), beam=float(processed[164][380:420, 300:500].mean()))


def compare(base, cand):
    fails = []
    for key, better, atol, rtol in CHECKS:
        b, c = base.get(key), cand.get(key)
        if b is None or c is None or b != b or c != c:
            continue
        tol = max(atol, rtol * abs(b))
        worse = (c > b + tol) if better == "lower" else (c < b - tol)
        if worse:
            fails.append(f"{key} {b:.3f} -> {c:.3f}")
    eb, ec = abs(base["exposure"] - 1), abs(cand["exposure"] - 1)
    if ec > eb + 0.01:
        fails.append(f"exposure {base['exposure']:.3f} -> {cand['exposure']:.3f}")
    if "flash_core" in base:
        for key in ("flash_core", "beam"):
            if abs(cand[key] - base[key]) > 0.01 * base[key] + 0.5:
                fails.append(f"range {key} {base[key]:.1f} -> {cand[key]:.1f}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--base", default="fusion")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every prepared benchmark clip")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    a.clips = a.clips or default_clips()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"gate_{a.candidate}.json")
    rep = json.load(open(path)) if os.path.exists(path) else {}
    for name in a.clips:
        meta = load_meta(name)
        orig = load_frames(name)
        scope = analyze(orig, name, meta["fps"])
        t0 = time.time()
        if not a.no_run or not has_method(name, a.candidate):
            save_method(name, a.candidate, METHODS[a.candidate](name, orig, meta, scope=scope))
        secs = time.time() - t0
        res = {}
        for m in (a.base, a.candidate):
            out = load_frames(name, m)
            r = score(name, meta, orig, out, scope)
            r = {k: v for k, v in r.items() if isinstance(v, (int, float))}
            r["artifact_pct"] = artifact_pct(orig, out)
            if name == "range":
                r.update(shot_checks(out))
            res[m] = r
        fails = compare(res[a.base], res[a.candidate])
        rep[name] = dict(base=res[a.base], cand=res[a.candidate], fails=fails, seconds=secs)
        json.dump(rep, open(path, "w"), indent=1)
        b, c = res[a.base], res[a.candidate]
        print(f"{'FAIL' if fails else 'pass'}  {name:24s} flicker {b['flicker_pct']:6.2f}->{c['flicker_pct']:6.2f}  "
              f"warp {b['ewarp']:6.2f}->{c['ewarp']:6.2f}  struct {b['structure']:.3f}->{c['structure']:.3f}  "
              f"artifact {b['artifact_pct']:5.2f}->{c['artifact_pct']:5.2f}%  {'; '.join(fails)}", flush=True)
        del orig, scope
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
