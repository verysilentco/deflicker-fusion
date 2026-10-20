"""Visual report for FlickerScope: where and when a clip flickers.

Writes to OUT/<clip>/:
  flickerscope.png  frame | Percent-Flicker heatmap | phase map | per-frame flicker curve
  cubes.png         grid of spatio-temporal cubes (regions x time segments), % flicker in each
  cubes.json        the same numbers
  flicker_overlay.mp4  the clip with the live flicker gain painted over it
"""
import json
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, load_frames, load_meta, write_video  # noqa: E402
from flickerscope import PERCEPTIBLE, analyze  # noqa: E402


def cube_table(logG, vis, rows, cols, segs):
    T, h, w = logG.shape
    G = torch.exp(logG)
    out = np.zeros((segs, rows, cols))
    edges_t = np.linspace(0, T, segs + 1).astype(int)
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    for s in range(segs):
        g = G[edges_t[s]:edges_t[s + 1]].reshape(edges_t[s + 1] - edges_t[s], -1)
        lo = torch.quantile(g, 0.02, dim=0).reshape(h, w)
        hi = torch.quantile(g, 0.98, dim=0).reshape(h, w)
        pct = ((hi - lo) / (hi + lo)).cpu().numpy()
        for r in range(rows):
            for c in range(cols):
                v = vis[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                p = pct[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                out[s, r, c] = (p * v).sum() / max(v.sum(), 1e-6) * 100 if v.sum() > 0.05 * v.size else np.nan
    return out, edges_t


def report(name, frames=None, label="original", f0=None):
    meta = load_meta(name)
    frames = load_frames(name) if frames is None else frames
    r = analyze(frames, name, meta["fps"], f0=f0)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    T, H, W, _ = frames.shape
    pct, vis, phase = r["maps"]["pct"], r["maps"]["vis"], r["maps"]["phase"]
    mid = frames[T // 2]
    portrait = H > W

    # ---- panel
    fig = plt.figure(figsize=(14, 7.5) if portrait else (16, 7.5), dpi=110)
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.2])
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(mid); ax.set_title(f"{name} — {label}"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(mid)
    pm = cv2.resize(pct * 100, (W, H), interpolation=cv2.INTER_LINEAR)
    vm = cv2.resize(vis, (W, H), interpolation=cv2.INTER_LINEAR)
    im = ax.imshow(np.ma.masked_where(vm < 0.2, pm), cmap="inferno", vmin=0, vmax=max(20, np.nanpercentile(pm, 99)),
                   alpha=0.75)
    ax.set_title(f"Percent Flicker per pixel  (video: {r['flicker_pct']:.1f}%)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% flicker (IEEE 1789)")
    ax = fig.add_subplot(gs[0, 2])
    ph = cv2.resize(phase, (W, H), interpolation=cv2.INTER_NEAREST)
    amp = cv2.resize(pct, (W, H))
    hsv = np.stack([(ph + np.pi) / (2 * np.pi) * 179, np.full_like(ph, 220),
                    np.clip(amp / max(1e-6, np.nanpercentile(amp, 98)), 0, 1) * 255], -1).astype(np.uint8)
    ax.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)); ax.axis("off")
    ax.set_title("Flicker phase (hue) × strength — bands = rolling shutter / separate lights")
    ax = fig.add_subplot(gs[1, :])
    curve = r["maps"]["frame_curve"]
    ax.plot((np.exp(curve) - 1) * 100, lw=1.2, color="#e8590c")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("frame"); ax.set_ylabel("frame brightness\nvs steady (%)"); ax.set_xlim(0, T - 1)
    ax.set_title(f"f0 = {r['f0']:.4f} cycles/frame  ({r['hz']:.2f} Hz on playback, period {r['period_frames']:.2f} frames,"
                 f" aliased components {', '.join(f'{c:.3f}' for c in r['components'])})  ·  coverage {r['coverage_pct']:.0f}%"
                 f"  ·  peak {r['peak_pct']:.0f}%  ·  Flicker Index {r['flicker_index']:.3f}", fontsize=9)
    fig.tight_layout()
    tag = "" if label == "original" else f"_{label}"
    fig.savefig(os.path.join(d, f"flickerscope{tag}.png"))
    plt.close(fig)

    # ---- cubes
    rows, cols = (8, 5) if portrait else (5, 8)
    segs = 4
    cubes, et = cube_table(r["logG"], vis, rows, cols, segs)
    fig, axs = plt.subplots(1, segs, figsize=(4.2 * segs, 6.8 if portrait else 3.6), dpi=110)
    for s in range(segs):
        ax = axs[s]
        ax.imshow(frames[(et[s] + et[s + 1]) // 2])
        ys = np.linspace(0, H, rows + 1); xs = np.linspace(0, W, cols + 1)
        for rr in range(rows):
            for cc in range(cols):
                v = cubes[s, rr, cc]
                if np.isnan(v):
                    txt, col = "dark", (0.5, 0.5, 0.5, 0.35)
                else:
                    a = min(1.0, v / 40.0)
                    col = (1.0, 0.25 + 0.5 * (1 - a), 0.1, 0.15 + 0.45 * a) if v >= PERCEPTIBLE * 100 else (0.2, 0.8, 0.4, 0.25)
                    txt = f"{v:.0f}%"
                ax.add_patch(plt.Rectangle((xs[cc], ys[rr]), xs[cc + 1] - xs[cc], ys[rr + 1] - ys[rr],
                                           facecolor=col, edgecolor="white", lw=0.6))
                ax.text((xs[cc] + xs[cc + 1]) / 2, (ys[rr] + ys[rr + 1]) / 2, txt, ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        ax.set_title(f"frames {et[s]}–{et[s + 1] - 1}"); ax.axis("off")
    fig.suptitle(f"{name} — {label}: flicker per spatio-temporal cube  (green < 3% = imperceptible, 'dark' = too dark to judge)")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"cubes{tag}.png"))
    plt.close(fig)
    with open(os.path.join(d, f"cubes{tag}.json"), "w") as f:
        json.dump(dict(rows=rows, cols=cols, segments=[[int(et[s]), int(et[s + 1])] for s in range(segs)],
                       pct=np.nan_to_num(cubes, nan=-1).round(2).tolist(),
                       summary={k: r[k] for k in ["flicker_pct", "coverage_pct", "peak_pct", "flicker_index",
                                                  "f0", "hz", "components", "periodic"]}), f, indent=1)

    # ---- overlay video: live gain, red = brighter than steady, blue = darker
    logG = r["logG"].cpu().numpy()
    ov = np.empty_like(frames)
    for t in range(T):
        g = cv2.resize(logG[t], (W, H), interpolation=cv2.INTER_LINEAR)
        a = np.clip(np.abs(g) / 0.5, 0, 1)[..., None] * 0.65
        tint = np.where(g[..., None] > 0, np.array([255, 60, 30]), np.array([30, 120, 255]))
        ov[t] = (frames[t] * (1 - a) + tint * a).astype(np.uint8)
    write_video(np.concatenate([frames, ov], axis=2), os.path.join(d, f"flicker_overlay{tag}.mp4"), meta["fps"])
    return r


if __name__ == "__main__":
    for n in sys.argv[1:]:
        res = report(n)
        print(n, f"{res['flicker_pct']:.1f}%", "->", os.path.join(OUT, n))
