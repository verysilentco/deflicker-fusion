"""Physically modelled synthetic lighting flicker with exact ground truth.

Takes clean frames (sRGB uint8) and relights them with mains-driven sources as a high-speed camera
would see them:

  light   L_j(t)   waveform of source j: incandescent / magnetic-ballast fluorescent / LED rectified /
                   LED PWM, at 2 x mains (50 or 60 Hz) or a PWM frequency, with a 3-phase offset
  camera  frame n of row r integrates the light over [n/fps + r*line_time, ... + exposure]
          (exposure averaging softens flicker; rolling shutter shifts its phase down the frame)
  scene   I(x, n) = S(x, n) * (a(x) + sum_j m_j(x) * L_j(n, row(x))) / mean light
          S = clean radiance, a = steady ambient share, m_j = smooth footprint of source j
  sensor  exposure scaling with highlight clipping, Poisson-Gaussian noise, sRGB encode, 8-bit

Ground truth returned: gain field G(x, n) (multiplicative illumination relative to its temporal mean),
per-pixel IEEE 1789 Percent Flicker of G, and the clean steady frames.
"""
import math

import numpy as np

from common import lin_to_srgb, srgb_to_lin

WAVEFORMS = ("incandescent", "fluorescent", "led_rectified", "led_pwm")


def waveform(kind, t, f_mains, rng_params):
    """Instantaneous relative light output (mean ~1) of a source at times t (seconds)."""
    phase = rng_params["phase"]
    if kind == "led_pwm":
        f = rng_params["pwm_hz"]
        duty = rng_params["duty"]
        return (((t * f + phase / (2 * math.pi)) % 1.0) < duty).astype(np.float64) / duty
    s = np.abs(np.sin(2 * math.pi * f_mains * t + phase))  # rectified mains: 2 x f_mains flicker
    if kind == "incandescent":
        depth = rng_params.get("depth", 0.12)  # filament thermal inertia: shallow, nearly sinusoidal
        return 1.0 + depth * (np.cos(4 * math.pi * f_mains * t + 2 * phase) * -1.0)
    if kind == "fluorescent":
        depth = rng_params.get("depth", 0.35)  # phosphor persistence leaves a floor + harmonics
        return (1 - depth) + depth * (s ** 1.5) / np.mean(np.abs(np.sin(np.linspace(0, math.pi, 1000))) ** 1.5)
    if kind == "led_rectified":
        return s / (2 / math.pi)  # unfiltered driver: full modulation, rectified sine
    raise ValueError(kind)


def integrate(kind, t0, exposure, f_mains, params, samples=24):
    """Average light over each exposure window [t0, t0 + exposure]."""
    u = (np.arange(samples) + 0.5) / samples
    ts = t0[..., None] + u * exposure
    return waveform(kind, ts, f_mains, params).mean(-1)


def smooth_field(H, W, rng, scale=0.35):
    """Smooth random footprint in [0,1] (sum of a few broad Gaussians)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    f = np.zeros((H, W))
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.uniform(-0.2, 1.2) * H, rng.uniform(-0.2, 1.2) * W
        sy, sx = rng.uniform(scale * 0.5, scale * 1.5) * H, rng.uniform(scale * 0.5, scale * 1.5) * W
        f += np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    return f / f.max()


def sample_config(rng, H, W):
    n_src = int(rng.integers(1, 4))
    fps = float(rng.choice([120.0, 240.0, 240.0, 480.0]))
    f_mains = float(rng.choice([50.0, 60.0]))
    srcs = []
    for j in range(n_src):
        kind = str(rng.choice(WAVEFORMS, p=[0.15, 0.3, 0.3, 0.25]))
        p = dict(phase=float(rng.choice([0.0, 2 * math.pi / 3, 4 * math.pi / 3]) + rng.normal(0, 0.05)))
        if kind == "led_pwm":
            p.update(pwm_hz=float(rng.choice([100, 120, 200, 240, 400, 1000, 2000]) * rng.uniform(0.98, 1.02)),
                     duty=float(rng.uniform(0.2, 0.9)))
        elif kind in ("incandescent", "fluorescent"):
            p["depth"] = float(rng.uniform(0.05, 0.2) if kind == "incandescent" else rng.uniform(0.15, 0.6))
        srcs.append(dict(kind=kind, params=p, footprint=smooth_field(H, W, rng), weight=float(rng.uniform(0.4, 1.0))))
    return dict(
        fps=fps, f_mains=f_mains, sources=srcs,
        exposure=float(rng.uniform(0.15, 1.0)) / fps,               # shutter as a fraction of the frame time
        line_time=float(rng.choice([0.0, rng.uniform(5e-6, 3e-5)])),  # 0 = global shutter
        ambient=float(rng.uniform(0.0, 0.5)),
        exposure_gain=float(rng.uniform(0.8, 1.6)),                  # >1 pushes highlights into clipping
        read_noise=float(rng.uniform(0.001, 0.006)), shot_noise=float(rng.uniform(0.0005, 0.004)),
    )


def apply(clean_srgb, cfg, rng, t_offset=0.0):
    """clean_srgb: (T,H,W,3) uint8. Returns (flickered uint8, G (T,H,W) float32, pct (H,W))."""
    T, H, W, _ = clean_srgb.shape
    S = srgb_to_lin(clean_srgb.astype(np.float64) / 255.0)
    rows = np.arange(H)[:, None] * np.ones((1, W))
    n = np.arange(T)[:, None, None]
    t0 = t_offset + n / cfg["fps"] + rows[None] * cfg["line_time"]          # (T,H,W) start of exposure
    num = cfg["ambient"] * np.ones((T, H, W))
    den = cfg["ambient"] * np.ones((1, H, W))
    for s in cfg["sources"]:
        # one light value per (frame, row): rolling shutter varies by row, columns share it
        L = integrate(s["kind"], t0[:, :, :1], cfg["exposure"], cfg["f_mains"], s["params"])
        L = np.broadcast_to(L, (T, H, W))
        m = s["weight"] * s["footprint"][None]
        num = num + m * L
        den = den + m
    illum = num / den                                                       # mean light per pixel ~ 1
    G = illum / illum.mean(0, keepdims=True)                                # GT gain vs temporal mean
    lin = S * G[..., None] * cfg["exposure_gain"]
    noisy = lin + rng.normal(0, 1, lin.shape) * np.sqrt(cfg["shot_noise"] * np.clip(lin, 0, None) + cfg["read_noise"] ** 2)
    out = np.clip(np.round(lin_to_srgb(np.clip(noisy, 0, 1)) * 255), 0, 255).astype(np.uint8)
    lo, hi = np.quantile(G, 0.02, axis=0), np.quantile(G, 0.98, axis=0)
    pct = (hi - lo) / (hi + lo)
    return out, G.astype(np.float32), pct.astype(np.float32)


def video_pct(pct, clean_srgb):
    """Ground-truth video Flicker % using FlickerScope's aggregation (visibility-weighted mean)."""
    Y = srgb_to_lin(clean_srgb.astype(np.float64) / 255).mean(-1).mean(0)
    vis = np.clip(Y / 0.01, 0, 1)
    return float((pct * vis).sum() / vis.sum() * 100)
