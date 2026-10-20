"""Write OUT/manifest.json for the testing-grounds viewer (viewer/index.html)."""
import glob
import json
import os

from common import OUT, list_clips, load_meta
from run import LABEL, ORDER


def build():
    clips = []
    for name in list_clips():
        d = os.path.join(OUT, name)
        if not os.path.isdir(d):
            continue
        meta = load_meta(name)
        mpath = os.path.join(d, "metrics.json")
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
        pairs = {os.path.basename(p)[5:-4]: f"out/{name}/{os.path.basename(p)}"
                 for p in glob.glob(os.path.join(d, "pair_*.mp4"))}
        methods = [m for m in ORDER if m in pairs] + sorted(m for m in pairs if m not in ORDER)
        maps = {}
        for tag, fn in [("original", ""), ("fusion", "_fusion")]:
            ov = os.path.join(d, f"flicker_overlay{fn}.mp4")
            if os.path.exists(ov):
                maps[tag] = dict(overlay=f"out/{name}/flicker_overlay{fn}.mp4",
                                 scope=f"out/{name}/flickerscope{fn}.png", cubes=f"out/{name}/cubes{fn}.png")
        clips.append(dict(
            name=name, note=meta.get("note", ""), fps=meta["fps"], frames=meta["frames"],
            width=meta["width"], height=meta["height"],
            methods=[dict(id=m, label=LABEL.get(m, m), video=pairs[m], metrics=metrics.get(m)) for m in methods],
            original=dict(metrics=metrics.get("original")),
            grid=f"out/{name}/grid.mp4" if os.path.exists(os.path.join(d, "grid.mp4")) else None,
            maps=maps,
        ))
    # your own clips first, then the dataset clips, then the synthetic ones
    clips.sort(key=lambda c: (c["name"].startswith("synth_"), c["name"].startswith("hf_"), c["name"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(dict(clips=clips), f, indent=1)
    return clips


if __name__ == "__main__":
    for c in build():
        print(c["name"], [m["id"] for m in c["methods"]])
