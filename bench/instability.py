"""Full-resolution temporal-instability map: how much each pixel jumps relative to its flow-aligned
neighbours (median of t-3..t+3), averaged over calm frames. Shows *where* residual flicker lives.

The __main__ block is the diagnostic used on the shooting-range clip ("range", 200 frames, shot at
frame 157; regions are crops of its 720x1280 frame): python instability.py -> OUT/range/instability.*
instability() itself works on any clip with a flow cache and a "fusion" output.
"""
import os, sys, numpy as np, torch, cv2
from common import DEV, FlowField, load_frames, srgb_to_lin, to_tensor, warp, OUT

@torch.no_grad()
def instability(name, frames, ff, ts, R=3):
    H, W = frames.shape[1:3]
    acc = torch.zeros(H, W, device=DEV); n = 0
    lum = lambda x: torch.log(srgb_to_lin(x).mul(torch.tensor([.2126,.7152,.0722], device=DEV)[None,:,None,None]).sum(1, keepdim=True) + 1e-3)
    for t in ts:
        nb = ff.neighbors(t, R)
        L0 = lum(to_tensor(frames[t:t+1]))
        vals = []
        for k in sorted(nb):
            if k == 0: continue
            fl, val = nb[k]
            Lk, ins = warp(lum(to_tensor(frames[t+k:t+k+1])), fl)
            vals.append(torch.where((val & ins)[:, None], Lk, torch.full_like(Lk, float('nan'))))
        med = torch.nanmedian(torch.cat(vals, 0), 0).values[0]
        d = (L0[0, 0] - med).abs()
        acc += torch.nan_to_num(d, nan=0.0); n += 1
    return (acc / n).cpu().numpy()

if __name__ == "__main__":
    name = "range"
    o = load_frames(name); f = load_frames(name, "fusion")
    H, W = o.shape[1:3]; ff = FlowField(name, H, W)
    ts = [t for t in range(4, 196) if not 145 <= t <= 185]
    mo = instability(name, o, ff, ts); mf = instability(name, f, ff, ts)
    os.makedirs(OUT + "/range", exist_ok=True)
    np.savez_compressed(OUT + "/range/instability.npz", orig=mo, fusion=mf)
    def heat(m, vmax=0.25):
        c = cv2.applyColorMap((np.clip(m / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    base = f[100]
    tiles = [base, heat(mo), heat(mf)]
    img = np.concatenate([cv2.resize(x, (360, 640)) for x in tiles], 1)
    cv2.imwrite(OUT + "/range/instability.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # regions from the screenshots (scaled from the 387x734 crop of the 720x1280 frame)
    for nm, (ys, xs) in {"red box (lane through glass)": (slice(490, 1150), slice(300, 640)),
                         "cap edge": (slice(380, 560), slice(60, 330)),
                         "lane, far from edges": (slice(620, 760), slice(430, 560)),
                         "shooter's back": (slice(800, 1100), slice(60, 250))}.items():
        print(f"{nm:30s} instability  original {mo[ys, xs].mean():.3f}   fusion {mf[ys, xs].mean():.3f}   fusion p99 {np.percentile(mf[ys, xs], 99):.3f}")
