"""Run the All-In-One-Deflicker (CVPR 2023) baseline on benchmark clips with its published defaults.

Saves out/aio (final, after local refinement) and out/aio_filter (neural filter only) per clip.
Runs inside your All-In-One-Deflicker checkout (DEFLICKER_RAFT, see common.py), with this Python
interpreter, so that checkout's own dependencies must be installed (see its README / environment.yml).

python aio_baseline.py hf_slowmo_face hf_slowmo_tv ...   (no arguments: every clip; ~1 h per 200 frames)
python aio_baseline.py --hybrid <clip> ...               ("AIO-improved": AIO's filter guided by Fusion)
"""
import json
import os
import shutil
import subprocess
import sys
import time

from common import AIO, clip_dir, frame_paths, has_method, list_clips, load_dir, load_meta, save_method

PY = sys.executable


def run(name, iters=10001):
    if has_method(name, "aio"):
        print(f"{name}: aio exists, skip", flush=True)
        return
    meta = load_meta(name)
    vid = f"bench_{name}"
    src = os.path.join(clip_dir(name), "frames")
    dst = os.path.join(AIO, "data", "test", vid)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    shutil.rmtree(os.path.join(AIO, "results", vid), ignore_errors=True)
    with open(os.path.join(AIO, "src", "config", "config_flow_100.json")) as f:
        cfg = json.load(f)
    cfg["iters_num"] = iters
    cfg["maximum_number_of_frames"] = max(200, meta["frames"])
    with open(os.path.join(AIO, "src", "config", "config_bench.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    env = dict(os.environ, PYTHONPATH=AIO)
    log = open(os.path.join(clip_dir(name), "aio.log"), "w")
    t0 = time.time()
    subprocess.run([PY, "src/stage1_neural_atlas.py", "--vid_name", vid, "--config", "config_bench.json",
                    "--down", "4", "--gpu", "0"], cwd=AIO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    t1 = time.time()
    subprocess.run([PY, "src/neural_filter_and_refinement.py", "--video_name", vid, "--fps",
                    str(int(round(meta["fps"])))], cwd=AIO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    t2 = time.time()
    res = os.path.join(AIO, "results", vid)
    final = load_dir(os.path.join(res, "final", "output"))
    filt = load_dir(os.path.join(res, "neural_filter", "output"))
    save_method(name, "aio", final)
    save_method(name, "aio_filter", filt)
    with open(os.path.join(clip_dir(name), "aio_timing.json"), "w") as f:
        json.dump(dict(atlas_s=t1 - t0, filter_s=t2 - t1, iters=iters, frames=len(final)), f)
    print(f"{name}: atlas {t1 - t0:.0f}s + filter {t2 - t1:.0f}s, {len(final)} frames", flush=True)


def run_hybrid(name, guide="fusion"):
    """AIO-improved: AIO's trained neural filter + local refinement, guided by our flicker-free
    estimate instead of a per-video neural atlas (so no atlas training at all)."""
    meta = load_meta(name)
    vid = f"bench_{name}_hyb"
    content = os.path.join(AIO, "data", "test", vid)
    styled = os.path.join(AIO, "results", vid, "stage_1", "output")
    shutil.rmtree(content, ignore_errors=True)
    shutil.rmtree(os.path.join(AIO, "results", vid), ignore_errors=True)
    shutil.copytree(os.path.join(clip_dir(name), "frames"), content)
    shutil.copytree(os.path.join(clip_dir(name), "out", guide), styled)
    env = dict(os.environ, PYTHONPATH=AIO)
    t0 = time.time()
    with open(os.path.join(clip_dir(name), "aio_hybrid.log"), "w") as log:
        subprocess.run([PY, "src/neural_filter_and_refinement.py", "--video_name", vid, "--fps",
                        str(int(round(meta["fps"])))], cwd=AIO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    res = os.path.join(AIO, "results", vid)
    save_method(name, "aio_hybrid", load_dir(os.path.join(res, "final", "output")))
    save_method(name, "aio_hybrid_filter", load_dir(os.path.join(res, "neural_filter", "output")))
    print(f"{name}: hybrid filter+refine {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--hybrid"]:
        for n in sys.argv[2:]:
            run_hybrid(n)
        sys.exit(0)
    for n in sys.argv[1:] or list_clips():
        try:
            run(n)
        except Exception as e:  # keep the queue going
            print(f"{n}: FAILED {e}", flush=True)
