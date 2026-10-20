"""Shared plumbing for the deflicker testing grounds.

Clips live under DATA/clips/<name>/frames/*.png with a meta.json; method outputs
under DATA/clips/<name>/out/<method>/*.png. Viewable videos go to OUT.
Frames are handled as RGB uint8 arrays shaped (T, H, W, 3).

Locations (defaults are relative to the repository root; each can be overridden):
  DEFLICKER_DATA  working data: clips, frames, flow caches, method outputs   (default ./data)
  DEFLICKER_OUT   rendered videos, maps, metrics, viewer manifest            (default ./out)
  DEFLICKER_RAFT  checkout of All-In-One-Deflicker, which provides the RAFT
                  code and pretrained_weights/raft-things.pth                 (default ./third_party/All-In-One-Deflicker)
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

# working data, outputs, and the RAFT code + weights; overridable via environment variables
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DATA = os.path.abspath(os.environ.get("DEFLICKER_DATA", os.path.join(ROOT, "data")))
OUT = os.path.abspath(os.environ.get("DEFLICKER_OUT", os.path.join(ROOT, "out")))
AIO = os.path.abspath(os.environ.get("DEFLICKER_RAFT", os.path.join(ROOT, "third_party", "All-In-One-Deflicker")))
FF = imageio_ffmpeg.get_ffmpeg_exe()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- clip I/O
def clip_dir(name):
    return os.path.join(DATA, "clips", name)


def list_clips():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, "clips", "*"))
                  if os.path.exists(os.path.join(p, "meta.json")))


def load_meta(name):
    with open(os.path.join(clip_dir(name), "meta.json")) as f:
        return json.load(f)


def frame_paths(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))


def load_dir(folder):
    return np.stack([cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                     for p in frame_paths(folder)])


def load_frames(name, method=None):
    sub = "frames" if method in (None, "original") else os.path.join("out", method)
    return load_dir(os.path.join(clip_dir(name), sub))


def save_dir(frames, folder):
    os.makedirs(folder, exist_ok=True)
    for old in frame_paths(folder):
        os.remove(old)
    for i, fr in enumerate(frames):
        cv2.imwrite(os.path.join(folder, f"{i:05d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])


def save_method(name, method, frames):
    save_dir(frames, os.path.join(clip_dir(name), "out", method))


def has_method(name, method):
    d = os.path.join(clip_dir(name), "out", method)
    return os.path.isdir(d) and len(frame_paths(d)) == load_meta(name)["frames"]


def write_video(frames, path, fps, crf=14):
    """RGB uint8 frames -> H.264 mp4 (even dims enforced)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T, H, W, _ = frames.shape
    H2, W2 = H - H % 2, W - W % 2
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W2}x{H2}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                          "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr[:H2, :W2]).tobytes())
    p.stdin.close()
    p.wait()


# ---------------------------------------------------------------- color
def srgb_to_lin(x):
    """x in [0,1] (sRGB/BT.709-ish transfer) -> linear light."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) if torch.is_tensor(x) \
        else np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    if torch.is_tensor(x):
        x = x.clamp(0, 1)
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luma_lin(lin_rgb):
    """Rec.709 relative luminance of linear RGB, channel-last or (…,3,H,W) tensors."""
    if torch.is_tensor(lin_rgb) and lin_rgb.dim() == 4 and lin_rgb.shape[1] == 3:
        return 0.2126 * lin_rgb[:, 0] + 0.7152 * lin_rgb[:, 1] + 0.0722 * lin_rgb[:, 2]
    return 0.2126 * lin_rgb[..., 0] + 0.7152 * lin_rgb[..., 1] + 0.0722 * lin_rgb[..., 2]


def to_tensor(frames):
    """(T,H,W,3) uint8 -> (T,3,H,W) float32 in [0,1] on DEV."""
    return torch.from_numpy(frames).to(DEV).permute(0, 3, 1, 2).float() / 255.0


def to_uint8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()


# ---------------------------------------------------------------- flow
_raft = None


def raft():
    global _raft
    if _raft is None:
        weights = os.path.join(AIO, "pretrained_weights", "raft-things.pth")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"RAFT weights not found at {weights}. Clone All-In-One-Deflicker into {AIO} (or point "
                f"DEFLICKER_RAFT at your checkout) and download its pretrained_weights; see THIRD_PARTY.md.")
        if AIO not in sys.path:
            sys.path.insert(0, AIO)
        from src.models.stage_1.core.raft import RAFT
        args = argparse.Namespace(small=False, mixed_precision=True, alternate_corr=False, dropout=0)
        m = torch.nn.DataParallel(RAFT(args))
        m.load_state_dict(torch.load(weights, map_location="cpu"))
        _raft = m.module.to(DEV).eval()
    return _raft


@torch.no_grad()
def raft_flow(a, b, iters=20):
    """a, b: (N,3,H,W) in [0,1]. Returns flow a->b in pixels, (N,2,H,W)."""
    H, W = a.shape[-2:]
    ph, pw = (-H) % 8, (-W) % 8
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    a8 = F.pad(a * 255.0, pad, mode="replicate")
    b8 = F.pad(b * 255.0, pad, mode="replicate")
    _, flow = raft()(a8, b8, iters=iters, test_mode=True)
    return flow[..., pad[2]:pad[2] + H, pad[0]:pad[0] + W].float()


def flow_path(name, tag=""):
    return os.path.join(clip_dir(name), f"flow{tag}.npz")


def load_flow(name, tag=""):
    """fw[t]: t -> t+1 on grid of t.  bw[t]: t+1 -> t on grid of t+1.  Stored at flow_scale."""
    z = np.load(flow_path(name, tag))
    return z["fw"], z["bw"], float(z["scale"])


def _grid(H, W, device):
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    return xx, yy


def warp(img, flow, mode="bilinear"):
    """Sample img (N,C,H,W) at x + flow (N,2,H,W). Returns (warped, inside_mask)."""
    N, C, H, W = img.shape
    xx, yy = _grid(H, W, img.device)
    x = xx + flow[:, 0]
    y = yy + flow[:, 1]
    grid = torch.stack([x / (W - 1) * 2 - 1, y / (H - 1) * 2 - 1], -1)
    out = F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    return out, inside


def fb_valid(fw, bw_next):
    """Forward-backward consistency for fw (t->t+1, grid t) and bw (t+1->t, grid t+1)."""
    bw_s, inside = warp(bw_next, fw)
    d = ((fw + bw_s) ** 2).sum(1)
    mag = (fw ** 2).sum(1) + (bw_s ** 2).sum(1)
    return (d < 0.01 * mag + 0.5) & inside


def resize_flow(flow, H, W):
    """Resize (N,2,h,w) flow to (H,W), scaling vectors."""
    h, w = flow.shape[-2:]
    f = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
    return torch.stack([f[:, 0] * (W / w), f[:, 1] * (H / h)], 1)


class FlowField:
    """Chained, occlusion-aware alignment of neighbours onto a reference frame.

    align(t, k) returns the flow from frame t to frame t+k (on t's grid) and a validity mask,
    built by composing consecutive RAFT flows; validity is the product of per-step
    forward-backward checks, so anything occluded along the way is excluded.
    """

    def __init__(self, name, H, W, tag=""):
        fw, bw, _ = load_flow(name, tag)
        self.fw = torch.from_numpy(fw)  # (T-1,2,h,w) half precision on CPU
        self.bw = torch.from_numpy(bw)
        self.H, self.W = H, W
        self.T = self.fw.shape[0] + 1
        self._cache = {}

    def _step(self, t, direction):
        """flow t -> t+direction on grid t at (H,W), plus its fb validity."""
        key = (t, direction)
        if key in self._cache:
            return self._cache[key]
        if direction > 0:
            f = self.fw[t:t + 1]
            b = self.bw[t:t + 1]
        else:
            f = self.bw[t - 1:t]
            b = self.fw[t - 1:t]
        f = resize_flow(f.to(DEV).float(), self.H, self.W)
        b = resize_flow(b.to(DEV).float(), self.H, self.W)
        v = fb_valid(f, b)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = (f, v)
        return f, v

    def neighbors(self, t, K):
        """{k: (flow t->t+k, valid)} for every k in [-K, K] inside the clip, built incrementally."""
        out = {0: self.align(t, 0)}
        for d in (1, -1):
            acc = valid = None
            cur = t
            for step in range(1, K + 1):
                if not 0 <= cur + d < self.T:
                    break
                f, v = self._step(cur, d)
                if acc is None:
                    acc, valid = f.clone(), v.clone()
                else:
                    f_s, ins = warp(f, acc)
                    v_s, _ = warp(v[:, None].float(), acc)
                    acc = acc + f_s
                    valid = valid & ins & (v_s[:, 0] > 0.5)
                cur += d
                out[d * step] = (acc, valid)
        return out

    def align(self, t, k):
        if k == 0:
            return torch.zeros(1, 2, self.H, self.W, device=DEV), torch.ones(1, self.H, self.W, dtype=torch.bool, device=DEV)
        d = 1 if k > 0 else -1
        acc, valid = self._step(t, d)
        acc = acc.clone()
        valid = valid.clone()
        cur = t + d
        for _ in range(abs(k) - 1):
            f, v = self._step(cur, d)
            f_s, ins = warp(f, acc)
            v_s, _ = warp(v[:, None].float(), acc)
            acc = acc + f_s
            valid = valid & ins & (v_s[:, 0] > 0.5)
            cur += d
        return acc, valid
