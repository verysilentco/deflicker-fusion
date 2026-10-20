"""Flickerformer (Qu et al., "It Takes Two", CVPR 2026) as a baseline, with its published inference.

Their test.py: frames [t-1, t, t+1] (edges: [t+1, t, t+2] / [t-1, t, t-2]), each resized to 512x512,
concatenated to 9 channels; the network output minus the resized frame t is resized back and added to
the full-resolution frame t. Reproduced here. The architecture file and the weights (Flickerformer.pth)
are loaded from your own checkout of https://github.com/qulishen/Flickerformer; neither is included in
this repository (see THIRD_PARTY.md). Checkout location: DEFLICKER_FLICKERFORMER, default
./third_party/Flickerformer.

python flickerformer_baseline.py hf_slowmo_girl hf_slowmo_face ...   (no arguments: every clip)
"""
import importlib.util
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, ROOT, has_method, list_clips, load_frames, save_method

REPO = os.path.abspath(os.environ.get("DEFLICKER_FLICKERFORMER", os.path.join(ROOT, "third_party", "Flickerformer")))
_model = None


def model():
    global _model
    if _model is None:
        if not os.path.isfile(os.path.join(REPO, "Flickerformer.pth")):
            raise FileNotFoundError(f"Flickerformer checkout with Flickerformer.pth not found at {REPO} "
                                    f"(set DEFLICKER_FLICKERFORMER); see THIRD_PARTY.md.")
        # minimal stand-in for basicsr.utils.registry so only the architecture file is imported
        reg = types.SimpleNamespace(register=lambda *a, **k: (lambda cls: cls))
        for mod in ("basicsr", "basicsr.utils", "basicsr.utils.registry"):
            sys.modules.setdefault(mod, types.ModuleType(mod))
        sys.modules["basicsr.utils.registry"].ARCH_REGISTRY = reg
        spec = importlib.util.spec_from_file_location("ff_arch", os.path.join(REPO, "basicsr", "archs", "Flickerformer_arch.py"))
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        net = arch.Flickerformer().to(DEV)
        sd = torch.load(os.path.join(REPO, "Flickerformer.pth"), map_location="cpu", weights_only=True)  # no pickle code
        sd = sd.get("params_ema", sd.get("params", sd))
        net.load_state_dict(sd)
        _model = net.eval()
    return _model


@torch.no_grad()
def run(frames):
    T, H, W, _ = frames.shape
    net = model()
    out = np.empty_like(frames)
    x = lambda i: torch.from_numpy(frames[i]).to(DEV).permute(2, 0, 1).float()[None] / 255.0
    r512 = lambda t: F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False, antialias=True)
    for t in range(T):
        if t == 0:
            a, b = 1, 2
        elif t == T - 1:
            a, b = T - 2, T - 3
        else:
            a, b = t - 1, t + 1
        cur = x(t)
        c = r512(cur)
        y = net(torch.cat([r512(x(a)), c, r512(x(b))], dim=1))
        res = F.interpolate(y - c, size=(H, W), mode="bilinear", align_corners=False)
        o = (cur + res).clamp(0, 1)
        out[t] = (o[0].permute(1, 2, 0) * 255 + 0.5).byte().cpu().numpy()
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or list_clips():
        if has_method(name, "flickerformer"):
            print(f"{name}: exists, skip", flush=True)
            continue
        t0 = time.time()
        out = run(load_frames(name))
        save_method(name, "flickerformer", out)
        print(f"{name}: flickerformer {time.time() - t0:.0f}s", flush=True)
