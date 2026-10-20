"""Validate FlickerScope against ground truth on synthetic clips (prep_synth.py).

For each synth_* clip: measured video Flicker % vs the true value, per-pixel Percent Flicker map
correlation and error vs the true map, and the detected frequency vs the physics.

python validate_scope.py            -> table + OUT/flickerscope_validation.json
"""
import json
import os

import cv2
import numpy as np

from common import OUT, clip_dir, list_clips, load_frames, load_meta
from flickerscope import METRIC, analyze


def main():
    rows = []
    for name in [c for c in list_clips() if c.startswith("synth_")]:
        meta = load_meta(name)
        gt = np.load(os.path.join(clip_dir(name), "gt.npz"))
        r = analyze(load_frames(name), name, meta["fps"], keep_gain=False, **METRIC)
        pm = r["maps"]["pct"]
        vis = r["maps"]["vis"]
        gt_map = cv2.resize(gt["pct"].astype(np.float32), (pm.shape[1], pm.shape[0]), interpolation=cv2.INTER_AREA)
        w = vis > 0.2
        corr = float(np.corrcoef(pm[w].ravel(), gt_map[w].ravel())[0, 1]) if w.sum() > 10 else float("nan")
        mae = float(np.abs(pm[w] - gt_map[w]).mean() * 100)
        cfg = meta["config"]
        # true flicker frequency in cycles/frame: 2 x mains (or the PWM rate) folded into [0, 0.5]
        fl = [s["params"]["pwm_hz"] if s["kind"] == "led_pwm" else 2 * cfg["f_mains"] for s in cfg["sources"]]
        true_f = sorted({round(abs(f / cfg["fps"] - round(f / cfg["fps"])), 4) for f in fl})
        rows.append(dict(clip=name, gt_pct=float(gt["video_pct"]), measured_pct=r["flicker_pct"],
                         map_corr=corr, map_mae_pts=mae, f0=r["f0"], true_f=true_f, periodic=r["periodic"],
                         sources=[s["kind"] for s in cfg["sources"]], shutter=cfg["exposure"] * cfg["fps"],
                         rolling=bool(cfg["line_time"])))
        x = rows[-1]
        print(f"{name}: true {x['gt_pct']:5.1f}%  measured {x['measured_pct']:5.1f}%  map r={corr:.2f}  "
              f"map MAE {mae:4.1f} pts  f0 {x['f0']:.3f} (true {true_f})  {', '.join(x['sources'])}", flush=True)
    if rows:
        g = np.array([x["gt_pct"] for x in rows]); m = np.array([x["measured_pct"] for x in rows])
        summ = dict(n=len(rows), mean_abs_err_pts=float(np.abs(g - m).mean()),
                    corr=float(np.corrcoef(g, m)[0, 1]) if len(rows) > 2 else float("nan"),
                    median_map_corr=float(np.nanmedian([x["map_corr"] for x in rows])))
        print(f"\nvideo Flicker %: mean |error| {summ['mean_abs_err_pts']:.2f} points, r = {summ['corr']:.3f};  "
              f"median per-pixel map r = {summ['median_map_corr']:.2f}")
        os.makedirs(OUT, exist_ok=True)
        json.dump(dict(summary=summ, clips=rows), open(os.path.join(OUT, "flickerscope_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
