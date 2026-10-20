"""Media for the results page (site/, published as the Hugging Face Space): web-sized before/after
videos, FlickerScope figures, a flash-frame comparison strip, and the exact points for the aliasing figure.
Reads the benchmark outputs (DATA clips, OUT metrics and renders); writes site/media/ and site/*.json.
Then render the page with make_page.py.

python build_site.py                          hero media for the shooting-range clip ("range", the author's own
                                              footage, not distributed; needs fusion, aio, ffmpeg, pp_current
                                              and scope_viz.py outputs for it)
python build_site.py --examples <clip> ...    before/after cards -> site/examples.json + media/ex_<clip>.*
python build_site.py --limits <clip> ...      "where it doesn't help yet" table -> site/limits.json
"""
import json
import math
import os

import cv2
import numpy as np

from common import OUT, ROOT, load_frames, load_meta, write_video

SITE = os.path.join(ROOT, "site")
MEDIA = os.path.join(SITE, "media")


def pair_video(name, method, out_name, half_w=432):
    meta = load_meta(name)
    o = load_frames(name)
    m = o if method == "original" else load_frames(name, method)
    T, H, W, _ = o.shape
    h = int(round(H * half_w / W / 2) * 2)
    frames = np.stack([np.concatenate([cv2.resize(o[t], (half_w, h), interpolation=cv2.INTER_AREA),
                                       cv2.resize(m[t], (half_w, h), interpolation=cv2.INTER_AREA)], 1)
                       for t in range(T)])
    write_video(frames, os.path.join(MEDIA, out_name), meta["fps"], crf=23)


def overlay_video(name, out_name, half_w=432):
    """original | live flicker gain painted over it (from scope_viz's overlay render)."""
    src = os.path.join(OUT, name, "flicker_overlay.mp4")
    cap = cv2.VideoCapture(src)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        H, W2 = f.shape[:2]
        h = int(round(H * half_w / (W2 // 2) / 2) * 2)
        frames.append(cv2.cvtColor(cv2.resize(f, (2 * half_w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    write_video(np.stack(frames), os.path.join(MEDIA, out_name), load_meta(name)["fps"], crf=23)


def flash_strip():
    o = load_frames("range")
    tiles = []
    for src, lab in [(o, "Original"), (load_frames("range", "aio"), "AIO (CVPR'23)"),
                     (load_frames("range", "fusion"), "Flicker Fusion")]:
        t = cv2.resize(src[157][280:720, 260:720], (460, 440), interpolation=cv2.INTER_AREA).copy()
        tiles.append(t)
    img = np.concatenate(tiles, 1)
    cv2.imwrite(os.path.join(MEDIA, "flash157.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def aliasing_points(fps=240.0, flicker_hz=100.0, frames=12, n=600):
    """Rectified-sine light output (50 Hz mains -> 100 Hz flicker) over 12 frames at 240 fps,
    the per-frame samples, and the exposure-averaged frame brightness, normalised to [0,1]."""
    T = frames / fps
    t = np.linspace(0, T, n)
    light = np.abs(np.sin(math.pi * flicker_hz * t))  # |sin(2*pi*50*t)| has a 100 Hz period
    st = (np.arange(frames) + 0.5) / fps
    samples = np.abs(np.sin(math.pi * flicker_hz * st))
    return dict(t=(t / T).round(5).tolist(), light=light.round(5).tolist(),
                st=(st / T).round(5).tolist(), samples=samples.round(5).tolist())


TITLES = {
    "hf_slowmo_rain": ("Water drops", "Falling water lit against black"),
    "hf_slowmo_girl": ("Night walkway", "Child running under street lamps"),
    "hf_slowmo_face": ("Close-up", "Laughing child, night lights behind"),
    "hf_slowmo_tv": ("Living room", "TV and window light"),
    "hf_slowmo_dog": ("Dog", "Dog crossing a hallway"),
    "hf_slowmo_india": ("Ceremony", "Stage lights at a wedding ceremony"),
    "hf_slowmo_bulb": ("Flame bulb", "Flame-effect lamp by a window"),
    "hf_slowmo_wedding": ("First dance", "Ballroom lighting, couple dancing"),
    "hf_slowmo_miss": ("Photographer", "Outdoor portrait, slow motion"),
    "hf_slowmo_fan": ("Ceiling fan", "Fan blades under a fluorescent tube"),
    "hf_slowmo_smoke": ("Smoker", "Low-key portrait, drifting smoke"),
    "hf_timelapse_sunset": ("Sunset", "Time-lapse exposure flicker"),
    "hf_timelapse_city": ("City", "Time-lapse exposure flicker"),
    "hf_timelapse_night2day": ("Snow day", "Time-lapse exposure flicker"),
    "hf_timelapse_snownight": ("Snow night", "Time-lapse exposure flicker"),
}


CAVEAT = {
    "hf_slowmo_rain": "Partial: the dark frames are lifted, but not all the way. Frame-to-frame brightness swing drops "
                      "from 2.8x to 1.8x, and the meter under-reads what is left because it books the deepest dips as "
                      "one-off events.",
    "hf_slowmo_fan": "Partial: brightness evens out, but the tube's colour still cycles white to yellow. The method "
                     "corrects brightness, not colour, so far.",
    "hf_slowmo_wedding": "Partial: the detected flicker is a slow swing of about 0.9 Hz, which is only partly removed.",
    "hf_slowmo_miss": "Partial: the detected flicker is a slow swing of about 1 Hz, which is only partly removed.",
}


def examples(clips, min_gain=0.5):
    """Web pair videos (original | Flicker Fusion), posters and a JSON index for the page's examples."""
    rows = []
    for c in clips:
        mp = os.path.join(OUT, c, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        o, f = m.get("original"), m.get("fusion")
        if not o or not f or o["flicker_pct"] - f["flicker_pct"] < min_gain:
            continue  # nothing to show: the clip had (almost) no flicker
        meta = load_meta(c)
        half = 360 if meta["height"] > meta["width"] else 480
        pair_video(c, "fusion", f"ex_{c}.mp4", half_w=half)
        poster = load_frames(c, "fusion")[meta["frames"] // 2]
        cv2.imwrite(os.path.join(MEDIA, f"ex_{c}.jpg"), cv2.cvtColor(cv2.resize(poster, (240, int(240 * meta["height"] / meta["width"]))),
                                                                   cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        t, note = TITLES.get(c, (c, ""))
        other = lambda r: dict(after=round(r["flicker_pct"], 1), warp=round(r["ewarp"], 1),
                               structure=round(r["structure"], 3)) if r and r.get("mver") == o.get("mver") else None
        rows.append(dict(id=c, title=t, note=note, fps=meta["fps"], frames=meta["frames"], w=meta["width"], h=meta["height"],
                         video=f"media/ex_{c}.mp4", poster=f"media/ex_{c}.jpg",
                         before=round(o["flicker_pct"], 1), after=round(f["flicker_pct"], 1),
                         warp_before=round(o["ewarp"], 1), warp_after=round(f["ewarp"], 1), structure=round(f["structure"], 3),
                         aio=other(m.get("aio")), ff=other(m.get("flickerformer")), caveat=CAVEAT.get(c)))
        print("example", c, rows[-1]["before"], "->", rows[-1]["after"], flush=True)
    json.dump(rows, open(os.path.join(SITE, "examples.json"), "w"), indent=1)
    return rows


WHY = {
    "hf_slowmo_fan": "The blades repeat every few frames. That periodic motion reads as flicker, and the method cannot separate it from the lamp.",
    "hf_slowmo_smoke": "Drifting smoke changes brightness with no lamp cycle, and flow cannot track it, so there is nothing reliable to fuse from.",
    "hf_timelapse_sunset": "Time-lapse: exposure jumps between shots taken seconds apart. There is no lamp cycle, and too much changes between frames to borrow light.",
    "hf_timelapse_city": "Time-lapse: exposure jumps between shots taken seconds apart. There is no lamp cycle, and too much changes between frames to borrow light.",
    "hf_timelapse_night2day": "Time-lapse: exposure jumps between shots taken seconds apart. There is no lamp cycle, and too much changes between frames to borrow light.",
    "hf_timelapse_snownight": "Time-lapse: exposure jumps between shots taken seconds apart. There is no lamp cycle, and too much changes between frames to borrow light.",
}


def limits(clips):
    """Clips where the method barely helps, with the reason: site/limits.json."""
    rows = []
    for c in clips:
        mp = os.path.join(OUT, c, "metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        o, f = m.get("original"), m.get("fusion")
        if not o or not f:
            continue
        t, note = TITLES.get(c, (c, ""))
        rows.append(dict(id=c, title=t, note=note, before=round(o["flicker_pct"], 1), after=round(f["flicker_pct"], 1),
                         why=WHY.get(c, "")))
        print("limit", c, rows[-1]["before"], "->", rows[-1]["after"], flush=True)
    json.dump(rows, open(os.path.join(SITE, "limits.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    import sys
    os.makedirs(MEDIA, exist_ok=True)
    if sys.argv[1:2] == ["--examples"]:
        examples(sys.argv[2:])
        sys.exit(0)
    if sys.argv[1:2] == ["--limits"]:
        limits(sys.argv[2:])
        sys.exit(0)
    for method, fn in [("fusion", "range_fusion.mp4"), ("aio", "range_aio.mp4"), ("ffmpeg", "range_ffmpeg.mp4"),
                       ("pp_current", "range_previous.mp4")]:
        pair_video("range", method, fn)
        print("wrote", fn, round(os.path.getsize(os.path.join(MEDIA, fn)) / 1e6, 2), "MB", flush=True)
    overlay_video("range", "range_map.mp4")
    print("wrote range_map.mp4", round(os.path.getsize(os.path.join(MEDIA, "range_map.mp4")) / 1e6, 2), "MB")
    flash_strip()
    for fn in ["flickerscope.png", "cubes.png"]:
        img = cv2.imread(os.path.join(OUT, "range", fn))
        s = 1400 / img.shape[1]
        cv2.imwrite(os.path.join(MEDIA, fn.replace(".png", ".jpg")), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
    json.dump(aliasing_points(), open(os.path.join(SITE, "aliasing.json"), "w"))
    print("done")
