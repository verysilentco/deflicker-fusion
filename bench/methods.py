"""Deflicker methods for the testing grounds. Each takes (name, frames, meta) and returns frames.

ffmpeg      ffmpeg's built-in `deflicker` filter (commodity baseline)
pp_current  our previous method: per-pixel temporal-median gain x flicker mask + guided denoise
scope_gain  ablation: divide by the FlickerScope gain field only (no fusion)
fusion      Flicker Fusion: FlickerScope-driven, SNR-weighted, motion-compensated fusion
(aio / aio_filter are produced by aio_baseline.py)
"""
import math
import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (DEV, FF, FlowField, clip_dir, frame_paths, lin_to_srgb, load_dir, srgb_to_lin, to_uint8,
                    to_tensor, warp)
from flickerscope import analyze, gauss_blur


# ---------------------------------------------------------------- ffmpeg
def m_ffmpeg(name, frames, meta):
    src = os.path.join(clip_dir(name), "frames")
    tmp = os.path.join(clip_dir(name), "_ffmpeg_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    first = os.path.basename(frame_paths(src)[0])
    subprocess.run([FF, "-loglevel", "error", "-framerate", str(meta["fps"]), "-start_number", first[:5].lstrip("0") or "0",
                    "-i", os.path.join(src, "%05d.png"), "-vf", "deflicker", "-start_number", "0",
                    os.path.join(tmp, "%05d.png")], check=True)
    out = load_dir(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---------------------------------------------------------------- previous method (PP_final)
def _movavg(x, w):
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), x], 0), 0)
    out = np.empty_like(x)
    for t in range(len(x)):
        a, b = max(0, t - w // 2), min(len(x), t + w // 2 + 1)
        out[t] = (c[b] - c[a]) / (b - a)
    return out


def _guided(I, p, r=7, eps=6.0):
    mI = cv2.boxFilter(I, -1, (r, r)); mp = cv2.boxFilter(p, -1, (r, r))
    cov = cv2.boxFilter(I * p, -1, (r, r)) - mI * mp
    var = cv2.boxFilter(I * I, -1, (r, r)) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return cv2.boxFilter(a, -1, (r, r)) * I + cv2.boxFilter(b, -1, (r, r))


def m_pp_current(name, frames, meta):
    fb = frames[..., ::-1].astype(np.float32)  # the original pipeline worked in BGR
    n, H, W, _ = fb.shape
    s = fb[:, ::4, ::4]
    L = 0.299 * s[..., 2] + 0.587 * s[..., 1] + 0.114 * s[..., 0]
    hs, ws = L.shape[1:]
    gm = L.reshape(n, -1).mean(1); gm = gm - gm.mean(); best = (3, 0)
    for p in range(3, 31):
        m = abs(np.sum(gm * np.exp(-1j * 2 * np.pi / p * np.arange(n)))) / n
        if m > best[1]:
            best = (p, m)
    period = best[0]; R = min(12, period)
    med = np.stack([np.median(L[max(0, t - R):min(n, t + R + 1)], axis=0) for t in range(n)])
    raw_gain = np.clip(med / np.maximum(L, 1e-3), 0.4, 3.0)
    TL = np.array([cv2.resize(L[i], (24, 40), interpolation=cv2.INTER_AREA) for i in range(n)]).reshape(n, -1)
    base = _movavg(TL, int(round(period * 1.5)))
    U, S, _ = np.linalg.svd((TL - base) - (TL - base).mean(0), full_matrices=False)
    ref = (U[:, 0] * S[0])[:, None]; Wf = int(round(period * 4))
    Lf = L.reshape(n, -1); pb = _movavg(Lf, int(round(period * 1.5)))
    pstr = (_movavg((Lf - pb) * ref, Wf) / np.maximum(_movavg(ref * ref, Wf), 1e-6) * ref).std(0)
    lo, hi = np.percentile(pstr, 50), np.percentile(pstr, 62)
    w_pix = cv2.resize(cv2.GaussianBlur(np.clip((pstr - lo) / (hi - lo + 1e-9), 0, 1).reshape(hs, ws), (0, 0), 1.2),
                       (W, H))[..., None]
    out = np.empty_like(frames)
    for i in range(n):
        gu = cv2.resize(raw_gain[i], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        y = np.clip(fb[i] * (1 + (gu - 1) * w_pix), 0, 255)
        b = (np.clip(gu[..., 0] - 1.0, 0, 1.0) * w_pix[..., 0])[..., None]
        den = np.stack([_guided(fb[i][..., c], y[..., c]) for c in range(3)], -1)
        out[i] = np.clip(y * (1 - b) + den * b, 0, 255).astype(np.uint8)[..., ::-1]
    return out


# ---------------------------------------------------------------- FlickerScope-driven methods
def box(x, r):
    return F.avg_pool2d(F.pad(x, [r, r, r, r], mode="replicate"), 2 * r + 1, stride=1)


@torch.no_grad()
def guided_upsample(logG_lo, guide_full, r_lo=2, eps=0.02):
    """Edge-aware upsampling of a low-res log-gain map, guided by a full-res log-luminance image.

    Fits logG ~ a * guide + b locally at low resolution, then evaluates at full resolution:
    sharp where flicker strength changes at an object edge (shooter vs lane), smooth elsewhere.
    """
    h, w = logG_lo.shape[-2:]
    H, W = guide_full.shape[-2:]
    g = F.interpolate(guide_full, size=(h, w), mode="area")
    p = logG_lo
    mg, mp = box(g, r_lo), box(p, r_lo)
    cov = box(g * p, r_lo) - mg * mp
    var = box(g * g, r_lo) - mg * mg
    a = cov / (var + eps)
    b = mp - a * mg
    a, b = box(a, r_lo), box(b, r_lo)
    A = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
    B = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
    return A * guide_full + B


@torch.no_grad()
def full_res_gain(frames, scope, r_lo=2, eps=0.02, name=None, adaptive=False):
    """(T,1,H,W) linear-light flicker gain at full resolution (fp16, on DEV).

    adaptive: where content moves fast (beyond ~1 px/frame, fully at 2.5) the gain's fine detail cannot
    have been measured (the per-pixel fit follows a turning head, not a lamp), so only its smooth,
    room-scale part is kept there. Lamp light is smooth across a surface anyway."""
    T, H, W, _ = frames.shape
    logG = scope["logG"][:, None]
    out = torch.empty(T, 1, H, W, device=DEV, dtype=torch.float16)
    ev = scope.get("event_mask")
    frame_ev = (ev.float().mean((1, 2)) > 0.25).cpu().numpy() if ev is not None else np.zeros(T, bool)
    calm = np.flatnonzero(~frame_ev)
    for t in range(T):
        # The guide supplies edges, so it must show the scene's surfaces, not a flash's light pattern:
        # on a frame-level event, borrow the guide from the nearest calm frame.
        tg = t if not frame_ev[t] or not len(calm) else int(calm[np.argmin(np.abs(calm - t))])
        lin = srgb_to_lin(to_tensor(frames[tg:tg + 1]))
        lum = (0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2])[:, None]
        g0g = F.interpolate(logG[tg:tg + 1], size=(H, W), mode="bilinear", align_corners=False)
        guide = torch.log(lum / torch.exp(g0g) + 1e-3)  # first-pass deflickered luminance: stable edges
        lg = guided_upsample(logG[t:t + 1], guide, r_lo, eps)
        if adaptive:
            if t == 0:
                ff = FlowField(name, H, W)
            sp = [ff.align(t, d)[0].norm(dim=1, keepdim=True) for d in (-1, 1) if 0 <= t + d < T]
            speed = sum(sp) / len(sp)
            u = gauss_blur(torch.clamp((speed - 1.0) / 1.5, 0, 1), 3.0)
            lg = (1 - u) * lg + u * gauss_blur(lg, W / 24.0)
        out[t] = torch.exp(lg).half()[0]
    return out


def m_scope_gain(name, frames, meta, scope=None):
    scope = scope or analyze(frames, name, meta["fps"])
    G = full_res_gain(frames, scope)
    out = np.empty_like(frames)
    for t in range(len(frames)):
        lin = srgb_to_lin(to_tensor(frames[t:t + 1]))
        out[t] = ((lin_to_srgb(lin / G[t:t + 1].float()).clamp(0, 1) * 255 + 0.5).byte()
                  .permute(0, 2, 3, 1).cpu().numpy()[0])
    return out


def noise_sigma_srgb(frames, floor=0.25 / 255):
    """Robust per-image noise sigma (sRGB units, 0..1) from the Laplacian of flat regions.

    Inclusive threshold and NaN-safe: clips with large perfectly flat areas (letterboxing, blown
    skies) have a 40th-percentile gradient of exactly 0, and a strict '<' would select nothing.
    """
    sig = []
    for t in range(0, len(frames), max(1, len(frames) // 12)):
        g = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=1)
        grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        flat = grad <= np.percentile(grad, 40)
        textured = flat & (grad > 0)  # exclude dead-flat (clipped / letterbox) pixels that carry no noise
        sel = textured if textured.sum() > 1000 else flat
        if sel.sum() > 100:
            sig.append(1.4826 * np.median(np.abs(lap[sel])) / math.sqrt(20))
    sig = [v for v in sig if np.isfinite(v)]
    return float(max(np.median(sig), floor)) if sig else floor



@torch.no_grad()
def refine_gain_hr(name, frames, scope, G, S1, clip_thr=0.97, log_eps=1e-4, guide_eps=0.01, ts=None, debug=None,
                   flow_tag="", strict=False, adaptive=False):
    """Full-resolution refinement of the flicker gain.

    The coarse gain (from FlickerScope's work-resolution analysis) cannot resolve thin or sharply
    bounded objects that flicker differently from their surroundings: a chain in front of the lane,
    the rim of a light fixture, a silhouette edge. For every full-res pixel, along flow-aligned
    neighbours t+k, the residual r_k = log Y_k - log S1_t - log G_coarse(t+k) is
        constant  (absorbs any error in the pass-1 steady estimate S1_t)
      + periodic  (at FlickerScope's aliased components: the gain error the coarse map missed)
      + noise.
    Unreliable samples (occluded, clipped, outlying, events) are imputed as "no correction", so the
    estimate shrinks to the coarse gain where evidence is weak. The periodic part at k = 0 is
    Wiener-shrunk against its noise level and smoothed edge-aware, guided by S1.
    """
    T, H, W, _ = frames.shape
    comps = scope["components"]
    K = int(scope.get("window", 6))
    ff = FlowField(name, H, W, flow_tag)
    sig_s = noise_sigma_srgb(frames)
    ev = scope.get("event_mask")
    ev = F.max_pool2d(ev.float()[:, None], 3, stride=1, padding=1) if ev is not None else None
    Gn = G.clone()
    for t in (range(T) if ts is None else ts):
        nb = ff.neighbors(t, K)
        ks = sorted(nb)
        s1 = srgb_to_lin(to_tensor(S1[t:t + 1]))
        logS1 = torch.log(0.2126 * s1[:, 0:1] + 0.7152 * s1[:, 1:2] + 0.0722 * s1[:, 2:3] + log_eps)
        rs, oks, sds = [], [], []
        for k in ks:
            fl, val = nb[k]
            x = to_tensor(frames[t + k:t + k + 1])
            if k != 0:
                x, ins = warp(x, fl)
                g, _ = warp(G[t + k:t + k + 1].float(), fl)
                ok = (val & ins)[:, None]
                if adaptive:
                    ok = ok & (fl.norm(dim=1, keepdim=True) < 2.0 * max(1.5, W / 400.0))
            else:
                g = G[t:t + 1].float()
                ok = torch.ones_like(g, dtype=torch.bool)
            lin = srgb_to_lin(x)
            lum = 0.2126 * lin[:, 0:1] + 0.7152 * lin[:, 1:2] + 0.0722 * lin[:, 2:3]
            ok = ok & (x.amax(1, keepdim=True) < clip_thr)
            if ev is not None:
                e = F.interpolate(ev[t + k:t + k + 1], size=(H, W), mode="bilinear", align_corners=False)
                if k != 0:
                    e, _ = warp(e, fl)
                ok = ok & (e < 0.05)
            slope = 2.4 / 1.055 * ((x + 0.055) / 1.055) ** 1.4
            if strict and k != 0:
                # a sample is evidence only where its edges sit where the steady frame's edges are:
                # a dancer passing through the window would otherwise be fitted as a gain correction
                sd_k = (sig_s * slope).mean(1, keepdim=True) / g + 1e-4
                ok = ok & (_grad_disagreement(lin / g, s1, sd_k, torch.full_like(sd_k, 1e-3)) < 0.5)
            sds.append(((sig_s * slope).mean(1, keepdim=True) / (lum + log_eps))[0, 0])
            rs.append((torch.log(lum + log_eps) - logS1 - torch.log(g))[0, 0])
            oks.append(ok[0, 0])
        r = torch.stack(rs)
        ok = torch.stack(oks)
        sd = torch.stack(sds)
        med = torch.where(ok, r, torch.full_like(r, float("nan"))).nanmedian(0).values
        med = torch.nan_to_num(med, nan=0.0)
        rt = torch.where(ok, r, med[None].expand_as(r))  # unusable samples imputed = no periodic correction
        # uniform-weight projection onto [1, k, cos, sin ...]; ridge on the harmonic terms only
        kk = torch.tensor(ks, device=DEV, dtype=torch.float32)
        cols = [torch.ones_like(kk), kk / K]
        for fa in comps:
            ang = 2 * math.pi * fa * (t + kk)
            cols += [torch.cos(ang), torch.sin(ang)]
        A = torch.stack(cols, 1)
        p = A.shape[1]
        ridge = torch.eye(p, device=DEV) * 0.5
        ridge[0, 0] = ridge[1, 1] = 1e-6
        P = torch.linalg.solve(A.T @ A + ridge, A.T)  # (p, n)
        # Outliers are judged against a first periodic fit, not the median: where the coarse gain is
        # badly wrong (a steady line in front of a 14x-swinging lane) the residual swings hugely, and
        # those are exactly the samples that carry the correction. Rejected samples are imputed with
        # the fit's prediction, which keeps the projection linear.
        th = torch.einsum("pn,nhw->phw", P, rt)
        pred = torch.einsum("np,phw->nhw", A, th)
        ok = ok & ((r - pred).abs() < 4 * sd + 0.10)
        rt = torch.where(ok, r, pred)
        at0 = torch.zeros(p, device=DEV)
        for j, fa in enumerate(comps):
            at0[2 + 2 * j] = math.cos(2 * math.pi * fa * t)
            at0[3 + 2 * j] = math.sin(2 * math.pi * fa * t)
        c = at0 @ P  # (n,) weights mapping residual samples to the periodic correction at frame t
        delta = torch.einsum("n,nhw->hw", c, rt)
        noise = torch.sqrt(torch.einsum("n,nhw->hw", c ** 2, sd ** 2)) * 1.5
        if debug is not None:
            debug.setdefault(t, {})["raw"] = delta.clone()
            debug[t]["noise"] = noise.clone()
            debug[t]["ok_frac"] = ok.float().mean(0).clone()
        delta = delta * delta ** 2 / (delta ** 2 + noise ** 2 + 1e-12)  # Wiener-like shrinkage
        if strict:
            # Flicker is periodic. Accept a correction only where the periodic fit explains most of the
            # residual's swing around its trend; motion and misalignment leave residuals that are not.
            th = torch.einsum("pn,nhw->phw", P, rt)
            per = torch.einsum("np,phw->nhw", A[:, 2:], th[2:])
            trend = torch.einsum("np,phw->nhw", A[:, :2], th[:2])
            r2 = (per ** 2).sum(0) / (((rt - trend) ** 2).sum(0) + (sd ** 2).sum(0) + 1e-9)
            delta = delta * torch.clamp((r2 - 0.3) / 0.3, 0, 1)
        if ev is not None:
            et = F.interpolate(ev[t:t + 1], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            delta = delta * (1 - torch.clamp(et / 0.35, 0, 1))
        if debug is not None:
            debug[t]["shrunk"] = delta.clone()
        delta = guided_full(delta[None, None], logS1, r=1, eps=guide_eps)  # 3x3: keeps 3-5 px structures
        if debug is not None:
            debug[t]["final"] = delta[0, 0].clone()
        Gn[t] = (G[t:t + 1].float() * torch.exp(delta)).half()[0]
    return Gn


def guided_full(p, I, r=2, eps=0.01):
    """Guided filter (He et al.) at full resolution: smooths p while keeping the edges of I."""
    mI, mp = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp) / (box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)

@torch.no_grad()
def m_fusion(name, frames, meta, scope=None, R=None, clip_thr=0.97, kappa=2.5, gain_tol=0.10, r_lo=2, eps=0.02,
             hr=True, iterate=False, guard=True, strict=False, adaptive=False):
    """Flicker Fusion.

    For each frame t, every neighbour t+k (|k| <= R) is warped onto t with occlusion-checked flow
    and divided by its own flicker gain, giving an estimate of the steady (flicker-free) scene at t.
    Estimates are merged with weights that combine:
      * signal-to-noise: a neighbour captured under bright light carries far less amplified noise
        than a trough frame brightened 5-10x, so weight ~ gain^2;
      * clipping: samples saturated in their own frame are rejected, so highlights blown in the
        bright frames are rebuilt from the dark frames that saw them unclipped;
      * robustness: a neighbour whose low-passed difference to frame t exceeds the noise model is
        down-weighted, so a moving subject or a muzzle flash never ghosts or gets averaged away.
    """
    scope = scope or analyze(frames, name, meta["fps"])
    if strict:
        # events that alternate with flicker are flicker (see FlickerScope osc_events)
        scope = analyze(frames, name, meta["fps"], f0=scope["f0"], periodic=scope["periodic"], osc_events=True)
    G = full_res_gain(frames, scope, r_lo, eps, name=name, adaptive=adaptive)
    out = _fuse(name, frames, scope, G, R, clip_thr, kappa, gain_tol, strict=strict, adaptive=adaptive)
    tag = ""
    if iterate:
        # Bootstrap: optical flow on flickering frames is unreliable exactly where flicker differs
        # between neighbours (a steady object against a pulsing background looks different every
        # frame). Re-estimate flow on the pass-1 deflickered frames and redo the analysis with it.
        from prep import build_flow
        tag = "_df"
        build_flow(name, frames=out, tag=tag, normalize=False, force=True)
        scope = analyze(frames, name, meta["fps"], f0=scope["f0"], periodic=scope["periodic"], log_eps=1e-4,
                        flow_tag=tag)
        G = full_res_gain(frames, scope, r_lo, eps, name=name, adaptive=adaptive)
        out = _fuse(name, frames, scope, G, R, clip_thr, kappa, gain_tol, tag, strict=strict, adaptive=adaptive)
    if hr and scope.get("components"):
        # full-resolution gain refinement against the current steady scene, then fuse again
        G = refine_gain_hr(name, frames, scope, G, out, clip_thr, flow_tag=tag, strict=strict, adaptive=adaptive)
        out = _fuse(name, frames, scope, G, R, clip_thr, kappa, gain_tol, tag, strict=strict, adaptive=adaptive)
    if guard:
        out = no_harm(frames, out)
        out = exposure_guard(frames, out)
    return out


def exposure_guard(original, processed, lo=0.85, hi=1.18):
    """Deflickering moves light between frames but keeps the clip's overall exposure (the gain is
    normalised to its arithmetic mean); every working clip lands within 1.00 +- 0.035. A result
    that brightens or darkens the whole clip by more than ~15% means the flicker model failed on
    this footage, and the honest output is the untouched original."""
    s = (slice(None), slice(None, None, 4), slice(None, None, 4))
    a = float(srgb_to_lin(original[s].astype(np.float32) / 255.0).mean())
    b = float(srgb_to_lin(processed[s].astype(np.float32) / 255.0).mean())
    ratio = b / max(a, 1e-6)
    if not lo <= ratio <= hi:
        print(f"  exposure guard: result is {ratio:.2f}x the original exposure -> model failed, keeping original",
              flush=True)
        return original.copy()
    return processed



def exact_period(comps, max_p=30, tol=0.03):
    """Smallest integer P (frames) such that every aliased component is a multiple of 1/P, i.e. the
    flicker repeats exactly every P frames (100 Hz at 240 fps -> 12). None if there is no such P."""
    if not comps:
        return None
    for P in range(2, max_p + 1):
        if all(abs(c * P - round(c * P)) < tol for c in comps):
            return P
    return None

@torch.no_grad()
def reconstruct_highlights(lin, x, thr=0.97):
    """Rebuild channels clipped in some but not all of a pixel's colour channels.

    Skin clips in red first, a lit wall in green: the other channels still hold the texture. Each
    clipped channel is estimated from the brightest unclipped one times the local colour ratio of
    nearby fully unclipped pixels (normalised Gaussian, widened where the clipped area is large),
    and never set below what the camera recorded. Fully clipped pixels are left as they are.
    lin: (n,3,H,W) linear light; x: the same frames in sRGB 0..1."""
    W = lin.shape[-1]
    clip_c = x >= thr
    any_c, all_c = clip_c.any(1, keepdim=True), clip_c.all(1, keepdim=True)
    part = any_c & ~all_c
    if not bool(part.any()):
        return lin
    ok = (~any_c).float()
    e = 1e-4
    out = lin.clone()
    s1, s2 = max(2.0, W / 96.0), max(4.0, W / 24.0)
    d1, d2 = gauss_blur(ok, s1), gauss_blur(ok, s2)
    for c in range(3):
        need = part & clip_c[:, c:c + 1]
        if not bool(need.any()):
            continue
        best = torch.zeros_like(lin[:, :1])
        bestu = torch.full_like(lin[:, :1], -1.0)
        for u in range(3):
            if u == c:
                continue
            L = torch.log((lin[:, c:c + 1] + e) / (lin[:, u:u + 1] + e)) * ok
            Ls = torch.where(d1 > 0.05, gauss_blur(L, s1) / d1.clamp_min(1e-6), gauss_blur(L, s2) / d2.clamp_min(1e-6))
            est = lin[:, u:u + 1] * torch.exp(Ls)
            usable = (~clip_c[:, u:u + 1]) & (lin[:, u:u + 1] > bestu)
            best = torch.where(usable, est, best)
            bestu = torch.where(usable, lin[:, u:u + 1], bestu)
        rec = torch.maximum(lin[:, c:c + 1], best)
        out[:, c:c + 1] = torch.where(need & (bestu >= 0), rec, lin[:, c:c + 1])
    return out


def _grad_disagreement(S, S0, sd, sd0):
    """Normalised disagreement (0 = same structure, ~1 = unrelated) between the edges of two linear
    images, from gradients of log luminance; noise enters the denominator so flat areas score ~0."""
    lum = lambda v: 0.2126 * v[:, 0:1] + 0.7152 * v[:, 1:2] + 0.0722 * v[:, 2:3]
    a, b = lum(S), lum(S0)
    La, Lb = torch.log(a + 1e-3), torch.log(b + 1e-3)
    gx = lambda L: F.pad(L[..., :, 1:] - L[..., :, :-1], (0, 1, 0, 0))
    gy = lambda L: F.pad(L[..., 1:, :] - L[..., :-1, :], (0, 0, 0, 1))
    diff = (gx(La) - gx(Lb)).abs() + (gy(La) - gy(Lb)).abs()
    mag = gx(La).abs() + gy(La).abs() + gx(Lb).abs() + gy(Lb).abs()
    noise = 3.0 * (sd / (a + 1e-3) + sd0 / (b + 1e-3))     # noise of a log-gradient, roughly
    return box(diff, 1) / (box(mag, 1) + noise + 1e-3)


def _fuse(name, frames, scope, G, R, clip_thr, kappa, gain_tol, flow_tag="", only_t=None, debug=None, strict=False,
          adaptive=False):
    T, H, W, _ = frames.shape
    if R is None:
        comps = scope.get("components") or []
        R = 3 if not comps else int(min(4, max(2, round(1.25 / max(comps)))) + 1)
    ff = FlowField(name, H, W, flow_tag)
    sig_s = noise_sigma_srgb(frames)
    # adaptive: optical flow is good to a fraction of the distance travelled, so a neighbour is trusted
    # where its content arrived from within ~D pixels (range: under 1 px; a turning head: 3-6 px)
    D = max(1.5, W / 400.0)
    trust = lambda fl: torch.exp(-0.5 * (fl.norm(dim=1, keepdim=True) / D) ** 2) if adaptive else 1.0
    clip_lin = float(srgb_to_lin(torch.tensor(clip_thr)))
    # events (muzzle flash, lamp switching on): FlickerScope's per-pixel mask, dilated a little
    ev = scope.get("event_mask")
    if ev is None:
        ev = torch.zeros(T, 8, 8, dtype=torch.bool, device=DEV)
    # Frame-level events: a muzzle flash lights the whole scene, strongly near the gun and faintly
    # far away. Per-pixel thresholds would split that gradient into patches, so a frame whose event
    # area is large is handled as a whole (additive model everywhere) and never used as a source.
    ev_area = ev.float().mean((1, 2))
    frame_event = (ev_area > 0.25).cpu().numpy()
    P_exact = exact_period(scope.get("components") or [])
    ev = F.max_pool2d(ev.float()[:, None], 3, stride=1, padding=1)
    out = np.empty_like(frames)
    for t in (range(T) if only_t is None else only_t):
        if frame_event[t]:
            continue  # rebuilt after the calm frames (see below)
        # soft-edged event mask (smoothstep of the upsampled coverage): no blocky seams
        ev_t = F.interpolate(ev[t:t + 1], size=(H, W), mode="bilinear", align_corners=False)
        ev_t = torch.clamp(ev_t / 0.35, 0, 1) ** 2 * (3 - 2 * torch.clamp(ev_t / 0.35, 0, 1))
        nb = ff.neighbors(t, R)
        Ss, Ws = [], []
        nb_num = torch.zeros(1, 3, H, W, device=DEV)
        nb_den = torch.zeros(1, 1, H, W, device=DEV)
        x0 = to_tensor(frames[t:t + 1])
        I0 = srgb_to_lin(x0)
        if adaptive:
            I0 = reconstruct_highlights(I0, x0, clip_thr)  # skin clipped in red keeps its texture
        S0 = I0 / G[t:t + 1].float()
        clipped0 = (x0.amax(1, keepdim=True) >= clip_thr).float()
        sd0 = (sig_s * 2.4 / 1.055 * ((x0 + 0.055) / 1.055) ** 1.4).mean(1, keepdim=True) / G[t:t + 1].float() + 1e-4
        S0_lo = box(S0, 4)
        hp0 = S0 - box(S0, 2)
        pm_clip = None
        if P_exact and t - P_exact >= 0 and not frame_event[t - P_exact]:
            flp, valp = ff.align(t, -P_exact)
            xpm, insp = warp(to_tensor(frames[t - P_exact:t - P_exact + 1]), flp)
            pm_clip = ((xpm.amax(1, keepdim=True) >= clip_thr - 0.03) & (valp & insp)[:, None]).float()
        for k in sorted(nb):
            fl, val = nb[k]
            x = to_tensor(frames[t + k:t + k + 1])
            if k != 0:
                x, ins = warp(x, fl)
                g, _ = warp(G[t + k:t + k + 1].float(), fl)
                valid = (val & ins).float()[:, None]
            else:
                g = G[t:t + 1].float()
                valid = torch.ones_like(g)
            unclipped = (x.amax(1, keepdim=True) < clip_thr).float()
            lin = srgb_to_lin(x)
            if adaptive and k == 0:
                # frame t's own pixel, with partially clipped channels rebuilt, is a real sample:
                # where no neighbour can be trusted (a moving subject) it is the answer
                lin = I0
                unclipped = (x.amin(1, keepdim=True) < clip_thr).float()
            S = lin / g
            # noise of S in linear light: sRGB noise mapped through the transfer slope, divided by gain
            slope = 2.4 / 1.055 * ((x + 0.055) / 1.055) ** 1.4
            sd = (sig_s * slope).mean(1, keepdim=True) / g + 1e-4
            snr_w = 1.0 / sd ** 2
            if k == 0:
                rob = torch.ones_like(g)
            else:
                # Two-band robustness test.
                #  brightness (9x9 mean): tolerant, and more so the larger the correction gap, because
                #    gain estimates are least certain where neighbour and frame t differ most;
                #  structure (high-pass): strict, forgiven only up to the noise level, because a
                #    structural mismatch means misalignment (a swinging chain, a moving hand).
                tol = 0.02 + gain_tol * (torch.log(g) - torch.log(G[t:t + 1].float())).abs()
                d_lo = (box(S, 4) - S0_lo).abs().mean(1, keepdim=True)
                z_lo = d_lo / ((sd + sd0) / 9.0 + tol * S0_lo.mean(1, keepdim=True))
                d_hi = box(((S - box(S, 2)) - hp0).abs(), 1).mean(1, keepdim=True)
                z_hi = d_hi / (0.3 * torch.sqrt(sd ** 2 + sd0 ** 2) + 0.01 * S0_lo.mean(1, keepdim=True) + 1e-4)
                rob = torch.exp(-0.5 * torch.clamp(z_lo - kappa, min=0) ** 2) * \
                    torch.exp(-0.5 * torch.clamp(z_hi - (kappa + 0.5), min=0) ** 2)
                if strict:
                    # A misaligned neighbour must be out, not merely down-weighted: a bright-phase
                    # neighbour carries up to ~10x frame t's SNR weight, so even rob = 0.1 leaks
                    # through as a visible ghost (a dancer's doubled legs).
                    gate = ((z_lo < kappa + 1.5) & (z_hi < kappa + 2.0)).float()
                    # Edges must sit where frame t's edges are. Compared as gradients of log luminance,
                    # which a gain error (smooth, multiplicative) barely touches, so this tests alignment
                    # only. Soft, motion-blurred edges a few pixels off pass the brightness and high-pass
                    # tests above, and averaging them smears a moving subject.
                    e_rel = _grad_disagreement(S, S0, sd, sd0)
                    ok = torch.clamp((0.6 - e_rel) / 0.25, 0, 1)
                    gate = gate * ok
                    rob = rob * (1 - F.max_pool2d(1 - gate, 3, stride=1, padding=1))
                    # How well this neighbour lines up *around* each pixel, judged on frame t's non-event
                    # pixels (an event's own light differs by definition). Static scene: ~1 everywhere.
                    rr2 = max(8, W // 32)
                    ne = (1 - ev_t) * valid
                    ring_k = box(ok * ne, rr2) / box(ne, rr2).clamp_min(1e-6)
                    aligned_k = torch.clamp((ring_k - 0.6) / 0.3, 0, 1) * (box(ne, rr2) > 1e-3).float()
                # Frame t is saturated here. A neighbour may stand in only if the clipping is explained
                # by flicker, i.e. re-lit to frame t's light level it would clip too. Otherwise the
                # saturation is an event (a muzzle flash) the neighbour never saw, and frame t stays.
                relit = (S * G[t:t + 1].float()).amax(1, keepdim=True)
                consistent = (relit >= 0.85 * clip_lin).float()
                if pm_clip is not None:
                    # the same lamp phase one cycle earlier clips here too: the clipping is flicker
                    consistent = torch.maximum(consistent, pm_clip)
                if strict:
                    # Inside a blown-out area there is nothing to check alignment against, so the
                    # neighbour must prove it lines up on the unclipped picture *around* the area.
                    # A static lamp passes; a turning cheek does not, and stays as frame t saw it.
                    rr = max(6, W // 40)
                    un0 = 1 - clipped0
                    num, den = box(rob * un0, rr), box(un0, rr)
                    ring = torch.where(den > 1e-3, num / den.clamp_min(1e-6), torch.ones_like(num))
                    consistent = consistent * torch.clamp((ring - 0.5) / 0.4, 0, 1)
                rob = rob * (1 - clipped0) + clipped0 * consistent
                # never borrow across events, and never import a neighbour's event into frame t
                ev_k = F.interpolate(ev[t + k:t + k + 1], size=(H, W), mode="bilinear", align_corners=False)
                ev_k, _ = warp(torch.clamp(ev_k / 0.35, 0, 1), fl)
                rob = rob * (1 - ev_t) * (1 - ev_k)
                # the steady scene as the neighbours saw it, used to remove only the lamps' share at events
                wn = valid * unclipped * snr_w * (1 - ev_k) * float(not frame_event[t + k]) * trust(fl)
                if strict:
                    # only neighbours that line up around the pixel are evidence of the steady scene;
                    # with none (a moving subject) the pixel gets the single-frame correction, and the
                    # full-res event test below has nothing to fire on
                    wn = wn * aligned_k
                nb_num = nb_num + S * wn
                nb_den = nb_den + wn
            w = valid * unclipped * snr_w * rob
            if k != 0:
                w = w * trust(fl)
            if k != 0 and frame_event[t + k]:
                w = w * 0
            if k == 0:
                w = w + 1e-6 * snr_w  # frame t always contributes a little (keeps geometry if all else fails)
            Ss.append(S)
            Ws.append(w)
        Wt = torch.stack(Ws)
        Sf = (torch.stack(Ss) * Wt).sum(0) / Wt.sum(0).clamp_min(1e-12)
        # where every sample was clipped, the steady value is at least the clip level / own gain
        allclip = (Wt.sum(0) < 1e-9).float()
        Sf = Sf * (1 - allclip) + S0 * allclip
        # Events add their own light (I = S*G + flash), so dividing by G would dim the flash with the
        # lamps. There, subtract only the lamps' flicker share, S_steady * (G - 1), and keep the rest.
        S_nb = torch.where(nb_den > 1e-9, nb_num / nb_den.clamp_min(1e-12), S0)
        Gt = G[t:t + 1].float()
        ev_out = (I0 - S_nb * (Gt - 1.0)).clamp_min(0)
        # Full-resolution event test (closes holes in the coarse mask): light far beyond what the
        # lamps alone could produce here (neighbours' steady scene x this frame's gain) is an event.
        lum = lambda v: 0.2126 * v[:, 0:1] + 0.7152 * v[:, 1:2] + 0.0722 * v[:, 2:3]
        lum3 = lambda v: v  # per-channel lamp estimates (kept as a hook for luminance-only variants)
        pred = lum(S_nb) * Gt
        margin = 0.25 * pred + 3 * sd0 * Gt + 2e-3
        ex = torch.clamp((lum(I0) - pred - margin) / margin, 0, 1)
        ex = ex * (nb_den > 1e-9).float()
        if P_exact and t - P_exact >= 0 and not frame_event[t - P_exact]:
            # real added light exceeds the same lamp phase one cycle earlier too; a gain-model error does not
            flp, valp = ff.align(t, -P_exact)
            xp, insp = warp(srgb_to_lin(to_tensor(frames[t - P_exact:t - P_exact + 1])), flp)
            ex_pm = torch.clamp((lum(I0) - lum(xp) - margin) / margin, 0, 1)
            ex = torch.where((valp & insp)[:, None], torch.minimum(ex, ex_pm), ex)
        ev_full = torch.maximum(ev_t, box(F.max_pool2d(ex, 5, stride=1, padding=2), 2))
        if frame_event[t]:
            ev_full = torch.ones_like(ev_full)
        Sf = Sf * (1 - ev_full) + ev_out * ev_full
        if debug is not None:
            debug[t] = dict(Ws=[x.clone() for x in Ws], ks=sorted(nb), nb_den=nb_den.clone(), ev_full=ev_full.clone(), ev_t=ev_t.clone(), Gt=Gt.clone(),
                            S_nb=S_nb.clone(), I0=I0.clone(), Sf=Sf.clone(), Wsum=Wt.sum(0).clone(),
                            clipped0=clipped0.clone(), allclip=allclip.clone(), ex=ex.clone(), ev_out=ev_out.clone())
        Sf = torch.where(torch.isfinite(Sf), Sf, I0)  # a numerical edge case must never become black video
        out[t] = ((lin_to_srgb(Sf).clamp(0, 1) * 255 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()[0])

    # Frame-level events (a flash lighting the whole scene) are rebuilt last, from their calm
    # neighbours' finished outputs: steady scene = flow-aligned average of the nearest calm fused
    # frames on each side; lamp light = median of the phase-matched frames t-P, t+P and the gain
    # model; the event's own light = whatever the frame has beyond the lamps.
    for t in (range(T) if only_t is None else only_t):
        if not frame_event[t]:
            continue
        I0 = srgb_to_lin(to_tensor(frames[t:t + 1]))
        steady, wsum = torch.zeros_like(I0), torch.zeros_like(I0[:, :1])
        sides = []
        for d in (-1, 1):
            j = t + d
            while 0 <= j < T and frame_event[j] and abs(j - t) <= R + 2:
                j += d
            if not (0 <= j < T) or frame_event[j]:
                continue
            fl, val = ff.align(t, j - t)
            xj, ins = warp(srgb_to_lin(to_tensor(out[j:j + 1])), fl)
            # chains out of a flash frame fail the forward-backward check on *appearance* (the flash),
            # not motion: where the motion is tiny the alignment is still trustworthy
            ok = (val | (fl.norm(dim=1) < 1.5)) & ins
            m = ok.float()[:, None] * trust(fl) / abs(j - t)
            steady, wsum = steady + xj * m, wsum + m
            sides.append((xj, ok.float()[:, None] * trust(fl)))
        S_single = (reconstruct_highlights(I0, to_tensor(frames[t:t + 1]), clip_thr) if adaptive else I0) / G[t:t + 1].float()
        S_st = torch.where(wsum > 0, steady / wsum.clamp_min(1e-9), S_single)
        if adaptive and sides:
            # where neither side's content stayed put, there is no trustworthy steady scene: this
            # frame's own single-frame correction (no averaged copies of a moving subject)
            c = torch.clamp(sum(o for _, o in sides), 0, 1)
            S_st = c * S_st + (1 - c) * S_single
        if strict and sides:
            # The steady scene behind a flash is trusted where the frames before and after it agree
            # (a static scene: the range shot). Where they disagree, people moved: use the side whose
            # edges line up with this frame, and where neither does, this frame's own single-frame
            # correction. Averaging two misplaced copies of a moving dancer and then adding the
            # flash's light on top is what draws a glowing double outline.
            x0s = to_tensor(frames[t:t + 1])
            sdt = (sig_s * 2.4 / 1.055 * ((x0s + 0.055) / 1.055) ** 1.4).mean(1, keepdim=True) / G[t:t + 1].float() + 1e-4
            match = [torch.clamp((0.6 - _grad_disagreement(x, I0, sdt, sdt)) / 0.25, 0, 1) * o for x, o in sides]
            if len(sides) == 2:
                (xa, oa), (xb, ob) = sides
                lum = lambda v: 0.2126 * v[:, 0:1] + 0.7152 * v[:, 1:2] + 0.0722 * v[:, 2:3]
                d_lum = box((torch.log(lum(xa) + 1e-3) - torch.log(lum(xb) + 1e-3)).abs(), 2)
                agree = torch.clamp((0.6 - _grad_disagreement(xa, xb, sdt, sdt)) / 0.25, 0, 1) * \
                    torch.clamp((0.25 - d_lum) / 0.15, 0, 1) * oa * ob
                pick = torch.where(match[0] >= match[1], xa, xb)
                q = torch.maximum(match[0], match[1])
            else:
                agree = torch.zeros_like(match[0])
                pick, q = sides[0][0], match[0]
            agree = box(agree, 2)
            S_st = agree * S_st + (1 - agree) * (q * pick + (1 - q) * S_single)
        # Lamp light during the event: the cycle *before* the event saw the same lamp phase and none of
        # the event's aftermath (smoke drifting in front of the lamps afterwards). Fall back to the cycle
        # after, then to the gain model, pixel by pixel.
        lamp = S_st * G[t:t + 1].float()
        if P_exact:
            for sft in (P_exact, -P_exact):  # later-assigned wins: t-P has priority over t+P
                j = t + sft
                if 0 <= j < T and not frame_event[j]:
                    fl, val = ff.align(t, sft)
                    xj, ins = warp(srgb_to_lin(to_tensor(frames[j:j + 1])), fl)
                    ok = (val | (fl.norm(dim=1) < 1.5)) & ins
                    lamp = torch.where(ok[:, None].expand_as(xj), xj, lamp)
        res = S_st + (I0 - lamp).clamp_min(0)
        out[t] = ((lin_to_srgb(res).clamp(0, 1) * 255 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()[0])
    return out


@torch.no_grad()
def no_harm(original, processed, R=6, lo=1.2, hi=2.0, strip=160):
    """Do-no-harm guard: never increase local temporal energy.

    Per pixel, over a +-R frame window, compares the temporal variance of (3x3-smoothed) log
    luminance in the processed clip against the original. Where processing made a pixel fluctuate
    more than it did originally (ratio > lo), the result is blended back toward the original,
    fully at ratio >= hi. Catches anything the gain model got wrong locally, e.g. a steady thin
    object in front of a flickering background inheriting the background's correction.
    """
    T, H, W, _ = original.shape
    out = processed.copy()

    def logl(fr):
        x = srgb_to_lin(to_tensor(fr))
        y = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        return box(torch.log(y + 1e-4), 1)[:, 0]

    def local_var(L):
        c1 = torch.cat([torch.zeros_like(L[:1]), L.cumsum(0)])
        c2 = torch.cat([torch.zeros_like(L[:1]), (L * L).cumsum(0)])
        idx = torch.arange(T, device=DEV)
        a, b = (idx - R).clamp(0, T), (idx + R + 1).clamp(0, T)
        n = (b - a).float()[:, None, None]
        m = (c1[b] - c1[a]) / n
        return ((c2[b] - c2[a]) / n - m * m).clamp_min(0)

    for y0 in range(0, H, strip):
        y1 = min(H, y0 + strip + 2)
        Lo = torch.cat([logl(original[i:i + 20, y0:y1]) for i in range(0, T, 20)])
        Lp = torch.cat([logl(processed[i:i + 20, y0:y1]) for i in range(0, T, 20)])
        ratio = (local_var(Lp) + 1e-4) / (local_var(Lo) + 1e-4)
        a = torch.clamp((ratio - lo) / (hi - lo), 0, 1)[:, None]
        a = box(F.max_pool2d(a, 3, stride=1, padding=1), 1)  # cover the rim of thin structures, soft edge
        for i in range(0, T, 20):
            o = srgb_to_lin(to_tensor(original[i:i + 20, y0:y1]))
            p = srgb_to_lin(to_tensor(processed[i:i + 20, y0:y1]))
            q = p * (1 - a[i:i + 20]) + o * a[i:i + 20]
            res = to_uint8(lin_to_srgb(q))
            n0 = 0 if y0 == 0 else 1
            out[i:i + 20, y0 + n0:y1] = res[:, n0:]
    return out


METHODS = {
    "ffmpeg": m_ffmpeg,
    "pp_current": m_pp_current,
    "scope_gain": m_scope_gain,
    "fusion": m_fusion,
    # candidate: Fusion with strict alignment rules (no partial leaks, blown-out areas rebuilt only
    # from neighbours that line up around them); becomes the default only if gate.py passes
    "fusion_strict": lambda name, frames, meta, scope=None: m_fusion(name, frames, meta, scope=scope, strict=True),
    # candidate: motion-adaptive Fusion (distance-limited borrowing, speed-limited gain detail)
    "fusion_adaptive": lambda name, frames, meta, scope=None: m_fusion(name, frames, meta, scope=scope, adaptive=True),
}
