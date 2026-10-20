"""Deflicker any video with FlickerScope + Flicker Fusion.

python deflicker_file.py path/to/clip.mov             -> clip_deflickered.mp4, clip_compare.mp4, clip_report.json
python deflicker_file.py clip.mov --max-frames 400      (first 400 frames only)
python deflicker_file.py clip.mov --adaptive            (motion-adaptive Fusion: for moving subjects)
python deflicker_file.py clip.mov --out results/        (write the outputs elsewhere; default: next to the input)

Intermediate frames and flow live in DATA/_job_<name>/ (DEFLICKER_DATA, default ./data) until the run ends.

Long videos run in overlapping chunks (default 200 frames, 24 overlap) so GPU memory stays bounded.
Each chunk's result is written to disk as soon as it is done, so an interrupted run resumes where it
stopped (use --fresh to start over). The final video is stitched (crossfading the overlaps) and encoded
one frame at a time, so memory stays flat however long the clip is. Source audio is copied untouched.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from common import DATA, FF, clip_dir, frame_paths, save_dir
from flickerscope import METRIC, analyze
from methods import m_fusion
from prep import build_flow


def probe_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def extract(path, folder, max_frames=None):
    """Frames as PNG plus each frame's exact presentation time (from ffmpeg, not OpenCV, whose
    frame rate can be wrong for phone slow-motion files)."""
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    cmd = [FF, "-hide_banner", "-loglevel", "info", "-i", path, "-map", "0:v:0"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    r = subprocess.run(cmd + ["-vf", "showinfo", "-vsync", "0", "-start_number", "0", os.path.join(folder, "%05d.png")],
                       check=True, stderr=subprocess.PIPE, text=True, errors="ignore")
    pts = [float(x.split("pts_time:")[1].split()[0]) for x in r.stderr.splitlines() if "pts_time:" in x]
    return pts


def read(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def make_chunk(name, frames, fps, note):
    d = clip_dir(name)
    shutil.rmtree(d, ignore_errors=True)
    save_dir(frames, os.path.join(d, "frames"))
    H, W = frames.shape[1:3]
    json.dump(dict(name=name, fps=fps, frames=len(frames), width=W, height=H, note=note),
              open(os.path.join(d, "meta.json"), "w"))
    build_flow(name)


def plan_chunks(T, size, overlap):
    if T <= size:
        return [(0, 0, T)]
    step = size - overlap
    spans = []
    s = 0
    while True:
        e = min(T, s + size)
        spans.append([s, e])
        if e >= T:
            break
        s += step
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < overlap + 24:  # fold a sliver into its neighbour
        spans.pop()
        spans[-1][1] = T
    return [(i, s, e) for i, (s, e) in enumerate(spans)]


def encode_exact(frame_dir, pts, out_path, crf=16):
    """Encode PNG frames with their original per-frame timestamps (variable frame rate)."""
    lst = os.path.join(frame_dir, "_frames.ffconcat")
    files = frame_paths(frame_dir)
    durs = np.diff(pts).tolist()
    durs.append(float(np.median(np.diff(pts))) if len(pts) > 1 else 1 / 30)
    lines = ["ffconcat version 1.0"]
    for p, d in zip(files, durs):
        lines += [f"file '{os.path.basename(p)}'", f"duration {d:.6f}"]
    lines.append(f"file '{os.path.basename(files[-1])}'")  # the concat demuxer needs the last file repeated
    with open(lst, "w") as f:
        f.write("\n".join(lines) + "\n")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-fps_mode", "vfr",
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", out_path], check=True)


class Encoder:
    """Streaming H.264 writer (frames pushed one at a time)."""

    def __init__(self, path, W, H, fps, crf=16):
        self.W, self.H = W - W % 2, H - H % 2
        self.p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{self.W}x{self.H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                                   "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def push(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame[:self.H, :self.W]).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


class Stages:
    """Prints '@stage <name>' for the Lab and, when the next stage starts, how long the last one took."""

    def __init__(self):
        self.name, self.t = None, None

    def __call__(self, name=None):
        if self.name:
            print(f"    {self.name}: {time.time() - self.t:.1f} s", flush=True)
        self.name, self.t = name, time.time()
        if name:
            print(f"@stage {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore saved progress and start over")
    ap.add_argument("--keep", action="store_true", help="keep the work folder afterwards")
    ap.add_argument("--adaptive", action="store_true", help="motion-adaptive Fusion (moving subjects, faces)")
    a = ap.parse_args()

    src = os.path.abspath(a.video)
    stem = os.path.splitext(os.path.basename(src))[0]
    out_dir = a.out or os.path.dirname(src)
    tag = "".join(ch if ch.isalnum() else "_" for ch in stem)[:40]
    t0 = time.time()
    work = os.path.join(DATA, f"_job_{tag}")
    inp = os.path.join(work, "input")
    state_path = os.path.join(work, "state.json")
    method = "fusion_adaptive" if a.adaptive else "fusion"
    stage = Stages()
    state = json.load(open(state_path)) if os.path.exists(state_path) and not a.fresh else {}
    # saved progress is reused only for the same frames *and* the same method
    if a.fresh or state.get("max_frames") != a.max_frames or state.get("method", "fusion") != method \
            or not frame_paths(inp):
        shutil.rmtree(work, ignore_errors=True)
        stage("reading frames")
        pts = extract(src, inp, a.max_frames)
        state = dict(max_frames=a.max_frames, method=method, chunks={}, pts=pts)
        json.dump(state, open(state_path, "w"))
    paths = frame_paths(inp)
    T = len(paths)
    pts = np.array(state.get("pts") or [])
    if len(pts) != T:  # fall back to the container's nominal rate
        pts = np.arange(T) / probe_fps(src)
    fps = (T - 1) / (pts[-1] - pts[0]) if T > 1 and pts[-1] > pts[0] else probe_fps(src)
    # timing that a constant rate cannot reproduce within 20 ms gets exact per-frame timestamps
    irregular = bool(np.abs((pts - pts[0]) - np.arange(T) / fps).max() > 0.02)
    chunks = plan_chunks(T, a.chunk, a.overlap)
    print(f"{stem}: {T} frames @ {fps:.2f} fps, {len(chunks)} chunk(s), {method}", flush=True)
    print(f"@progress 0/{len(chunks)}", flush=True)

    for ci, s, e in chunks:
        cdir = os.path.join(work, f"out_c{ci:03d}")
        if str(ci) in state["chunks"] and len(frame_paths(cdir)) == e - s:
            r = state["chunks"][str(ci)]
            print(f"  chunk {ci}: frames {s}-{e - 1} already done ({r['flicker_before']:.1f}% -> {r['flicker_after']:.1f}%)", flush=True)
            print(f"@progress {ci + 1}/{len(chunks)}", flush=True)
            continue
        fr = np.stack([read(p) for p in paths[s:e]])
        name = f"_user_{tag}_c{ci:03d}"
        part = f"part {ci + 1} of {len(chunks)}: " if len(chunks) > 1 else ""
        stage(f"{part}tracking motion")
        make_chunk(name, fr, fps, f"{stem} frames {s}-{e - 1}")
        stage(f"{part}measuring flicker")
        scope = analyze(fr, name, fps)
        stage(f"{part}removing flicker")
        res = m_fusion(name, fr, dict(fps=fps), scope=scope, adaptive=a.adaptive)
        stage(f"{part}checking the result")
        before = analyze(fr, name, fps, keep_gain=False, **METRIC)
        after = analyze(res, name, fps, keep_gain=False, f0=before["f0"], periodic=before["periodic"], **METRIC)
        stage()
        save_dir(res, cdir)  # checkpoint: an interrupted run resumes after this chunk
        state["chunks"][str(ci)] = dict(frames=[s, e], flicker_before=before["flicker_pct"],
                                        flicker_after=after["flicker_pct"], hz=scope["hz"], f0=scope["f0"],
                                        periodic=scope["periodic"], events=[s + x for x in scope["events"]])
        json.dump(state, open(state_path, "w"), indent=1)
        print(f"  chunk {ci}: frames {s}-{e - 1}  flicker {before['flicker_pct']:.1f}% -> {after['flicker_pct']:.1f}%  "
              f"({scope['hz']:.2f} Hz on playback, {'periodic' if scope['periodic'] else 'aperiodic'})", flush=True)
        print(f"@progress {ci + 1}/{len(chunks)}", flush=True)
        if not a.keep:
            shutil.rmtree(clip_dir(name), ignore_errors=True)
        del fr, res

    # stitch frame by frame: crossfade over the overlaps, encode as we go
    stage("encoding video")
    def weight(ci, s, e, t):
        w = 1.0
        ov = a.overlap
        if s > 0 and t < s + ov:
            w = min(w, (t - s + 1) / (ov + 1))
        if e < T and t >= e - ov:
            w = min(w, (e - t) / (ov + 1))
        return w

    os.makedirs(out_dir, exist_ok=True)
    first = read(paths[0])
    H, W = first.shape[:2]
    silent = os.path.join(work, "_video.mp4")
    final_dir = os.path.join(work, "final")
    os.makedirs(final_dir, exist_ok=True)
    enc = None if irregular else Encoder(silent, W, H, fps)
    cmp_enc = Encoder(os.path.join(out_dir, f"{stem}_compare.mp4"), 2 * (W - W % 2), H, fps, crf=18)
    for t in range(T):
        acc, ws = None, 0.0
        for ci, s, e in chunks:
            if s <= t < e:
                w = weight(ci, s, e, t)
                img = read(os.path.join(work, f"out_c{ci:03d}", f"{t - s:05d}.png")).astype(np.float32)
                acc = img * w if acc is None else acc + img * w
                ws += w
        frame = np.clip(acc / max(ws, 1e-6) + 0.5, 0, 255).astype(np.uint8)
        if enc is None:
            cv2.imwrite(os.path.join(final_dir, f"{t:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            enc.push(frame)
        w2 = W - W % 2
        cmp_enc.push(np.concatenate([read(paths[t])[:, :w2], frame[:, :w2]], axis=1))
    if enc is None:
        encode_exact(final_dir, pts, silent)
    else:
        enc.close()
    cmp_enc.close()

    stage()
    final = os.path.join(out_dir, f"{stem}_deflickered.mp4")
    trim = ["-shortest"] if a.max_frames else []  # a partial run gets matching partial audio
    r = subprocess.run([FF, "-y", "-loglevel", "error", "-i", silent, "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
                        "-c", "copy", *trim, final])
    if r.returncode != 0:  # no usable audio: ship the video alone
        shutil.copy(silent, final)
    reps = [state["chunks"][str(ci)] for ci, _, _ in chunks]
    span = lambda r_: r_["frames"][1] - r_["frames"][0]
    fb = sum(span(r_) * r_["flicker_before"] for r_ in reps) / sum(span(r_) for r_ in reps)
    fa = sum(span(r_) * r_["flicker_after"] for r_ in reps) / sum(span(r_) for r_ in reps)
    rep = dict(source=src, method=method, frames=T, fps=fps, exact_timing=irregular, flicker_before_pct=fb, flicker_after_pct=fa,
               chunks=reps, seconds_this_run=time.time() - t0)
    json.dump(rep, open(os.path.join(out_dir, f"{stem}_report.json"), "w"), indent=1)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nFlicker {fb:.1f}% -> {fa:.1f}%   ({T} frames, {rep['seconds_this_run'] / 60:.1f} min this run)")
    print("wrote", final)
    print("wrote", os.path.join(out_dir, f"{stem}_compare.mp4"))


if __name__ == "__main__":
    main()
