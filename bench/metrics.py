"""Scores for a processed clip against its original.

flicker_pct / coverage_pct / peak_pct  residual flicker, FlickerScope at the original's f0 (lower = better)
ewarp        warping error x1000: mean |O_t - warp(O_t+1)| over non-occluded pixels, sRGB (lower = better;
             the field-standard temporal-consistency metric, independent of our flicker model)
structure    SSIM structure term vs the original, contrast- and brightness-invariant (higher = better;
             drops with ghosting, blur, halos, artefacts)
noise_ratio  grain inside the strongly flickering zone, relative to that zone in the original's
             brightest-lit frames (1.0 = as clean as under good light; >1 = added grain)
exposure     mean luminance vs the original (1.0 = same overall brightness)
"""
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, FlowField, to_tensor, warp
from flickerscope import METRIC, analyze


def _luma(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]


def _gwin(sigma=1.5, r=5):
    g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


@torch.no_grad()
def structure_sim(a, b):
    """Mean SSIM structure term s = (sxy + C3) / (sx sy + C3) between luma images (N,1,H,W) in [0,1]."""
    w = _gwin()
    mu_a, mu_b = F.conv2d(a, w, padding=5), F.conv2d(b, w, padding=5)
    va = (F.conv2d(a * a, w, padding=5) - mu_a ** 2).clamp_min(0)
    vb = (F.conv2d(b * b, w, padding=5) - mu_b ** 2).clamp_min(0)
    cab = F.conv2d(a * b, w, padding=5) - mu_a * mu_b
    C3 = (0.03 ** 2) / 2
    s = (cab + C3) / (va.sqrt() * vb.sqrt() + C3)
    return float(s[..., 6:-6, 6:-6].mean())


@torch.no_grad()
def ewarp(name, frames):
    T, H, W, _ = frames.shape
    ff = FlowField(name, H, W)
    tot, n = 0.0, 0
    for t in range(T - 1):
        fl, val = ff.align(t, 1)
        a = to_tensor(frames[t:t + 1])
        b, ins = warp(to_tensor(frames[t + 1:t + 2]), fl)
        m = (val & ins).float()[:, None]
        tot += float(((a - b).abs().mean(1, keepdim=True) * m).sum() / m.sum().clamp_min(1))
        n += 1
    return tot / n * 1000


def _flat_noise(gray, mask):
    """Trimmed std of the high-pass residual over the flattest half of the mask (continuous, so it
    is not fooled by 8-bit quantisation the way a median of Laplacians is)."""
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    grad = cv2.GaussianBlur(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1)),
                            (0, 0), 2.0)
    sel = mask & (grad < np.percentile(grad[mask], 50))
    if sel.sum() < 500:
        return float("nan")
    v = hp[sel]
    v = v[np.abs(v) < np.percentile(np.abs(v), 99)]
    return float(v.std())


def flicker_zone_noise(original, processed, pct_map, thr=0.2):
    """Grain where the flicker was: noise of the processed clip inside the strongly flickering zone,
    relative to the original's brightest-lit frames of that same zone (1.0 = as clean as good light)."""
    T, H, W, _ = original.shape
    mask = cv2.resize(pct_map.astype(np.float32), (W, H)) > thr
    if mask.sum() < 2000:
        return float("nan")
    zone_lum = np.array([original[t][mask].mean() for t in range(T)])
    bright = np.argsort(zone_lum)[-max(3, T // 4):]
    ref = np.nanmedian([_flat_noise(cv2.cvtColor(original[t], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, mask)
                        for t in bright])
    idx = range(0, T, max(1, T // 50))
    out = np.nanmedian([_flat_noise(cv2.cvtColor(processed[t], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, mask)
                        for t in idx])
    return float(out / max(ref, 1e-9))


def gt_target(name, meta):
    """Ground truth for synthetic clips: the clean scene under the *average* light level
    (clean radiance x the simulated exposure gain), in sRGB uint8."""
    from common import clip_dir, load_dir, lin_to_srgb, srgb_to_lin
    clean = load_dir(os.path.join(clip_dir(name), "gt")).astype(np.float64) / 255
    lin = srgb_to_lin(clean) * meta["config"]["exposure_gain"]
    return (np.clip(lin_to_srgb(np.clip(lin, 0, 1)), 0, 1) * 255 + 0.5).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))


@torch.no_grad()
def score(name, meta, original, processed, scope_orig=None):
    scope_orig = analyze(original, name, meta["fps"], keep_gain=False, **METRIC)
    # measure the processed clip at the original's frequency *and* periodic/aperiodic decision
    same = dict(f0=scope_orig["f0"], periodic=scope_orig["periodic"])
    r = analyze(processed, name, meta["fps"], keep_gain=False, **same, **METRIC)
    # calm-segment flicker: the same measurement with the original's event episodes (+-8 frames) left out,
    # so a method is not rewarded for smoothing away a real event or penalised for preserving it
    T = len(original)
    evf = set()
    for e in scope_orig.get("events", []):
        evf.update(range(max(0, e - 8), min(T, e + 9)))
    calm = [t for t in range(T) if t not in evf]
    rc = analyze(processed, name, meta["fps"], keep_gain=False, score_frames=calm, **same, **METRIC) \
        if 30 < len(calm) < T else r
    idx = range(0, T, max(1, T // 40))
    s = np.mean([structure_sim(F.avg_pool2d(_luma(to_tensor(original[i:i + 1])), 2),
                               F.avg_pool2d(_luma(to_tensor(processed[i:i + 1])), 2)) for i in idx])
    expo = float(np.mean([processed[i].mean() for i in idx]) / max(1e-6, np.mean([original[i].mean() for i in idx])))
    extra = {}
    if meta.get("synthetic"):
        tgt = gt_target(name, meta)
        extra = dict(psnr_gt=float(np.mean([psnr(processed[i], tgt[i]) for i in range(T)])))
    return dict(
        **extra,
        flicker_pct=r["flicker_pct"], coverage_pct=r["coverage_pct"], peak_pct=r["peak_pct"],
        flicker_calm_pct=rc["flicker_pct"], calm_frames=len(calm),
        flicker_index=r["flicker_index"], ewarp=ewarp(name, processed), structure=float(s),
        noise_ratio=flicker_zone_noise(original, processed, scope_orig["maps"]["pct"]), exposure=expo,
    )
