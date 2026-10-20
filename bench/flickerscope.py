"""FlickerScope: a no-reference, physically grounded flicker meter for video.

Model, in linear light and along motion trajectories:

    Y(x, t) = S(x, t) * G(x, t)

S is the steady scene radiance (changes only slowly or through motion), G is the illumination
flicker gain. log G is estimated per pixel and per frame as

    periodic  : Fourier series at the detected flicker fundamental f0 (up to 3 harmonics),
                fitted jointly with a local linear trend over flow-aligned neighbours (robust IRLS)
    aperiodic : the spatially smooth part of what the fit leaves over. Illumination changes are
                smooth across the image, sensor noise is not, so this separates flicker from grain.

Harmonic amplitudes are debiased against the residual noise floor, so a clean video reads ~0%.

Per pixel we report IEEE 1789 Percent Flicker of G, (Gmax - Gmin) / (Gmax + Gmin), with robust
percentiles over time, and the Flicker Index (area above the mean / total area). The video score
aggregates these with a visibility weight that ignores near-black and clipped pixels.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, FlowField, luma_lin, srgb_to_lin, to_tensor, warp

WORK_EDGE = 320
LOG_EPS = 1e-4
METRIC = dict(work_edge=320, log_eps=1e-4)  # frozen settings when FlickerScope is used as a metric
PERCEPTIBLE = 0.03  # 3% modulation: coverage threshold
MAX_AMP = 4.0  # total harmonic amplitude of log G, per pixel


def work_size(H, W, edge=WORK_EDGE):
    s = edge / max(H, W)
    return max(8, int(round(H * s))), max(8, int(round(W * s)))


def moving_avg(x, w):
    """Centered moving average along dim 0 with edge renormalisation. x: (T, N)."""
    T = x.shape[0]
    c = torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)])
    idx = torch.arange(T, device=x.device)
    lo = (idx - w // 2).clamp(0, T)
    hi = (idx + w // 2 + 1).clamp(0, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None].float()


def gauss_blur(x, sigma):
    """Separable Gaussian blur of (N,C,H,W) with reflect padding."""
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, [r, r, 0, 0], mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, [0, 0, r, r], mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


@torch.no_grad()
def lin_luma_stack(frames, h, w):
    """uint8 (T,H,W,3) -> linear luminance (T,1,h,w) and a clipped-footprint mask (T,1,h,w)."""
    ys, cl = [], []
    for i in range(0, len(frames), 16):
        x = to_tensor(frames[i:i + 16])
        clipped = (x.amax(1, keepdim=True) > 0.975).float()
        y = luma_lin(srgb_to_lin(x))[:, None]
        ys.append(F.interpolate(y, size=(h, w), mode="area"))
        cl.append(F.interpolate(clipped, size=(h, w), mode="area") > 0.02)
    return torch.cat(ys), torch.cat(cl)


@torch.no_grad()
def detect_fundamental(logY, min_period=2.0, max_period=25.0):
    """Dominant periodic flicker frequency in cycles/frame, from whitened tile spectra."""
    T = logY.shape[0]
    tiles = F.adaptive_avg_pool2d(logY, (12, 12)).reshape(T, -1)
    s = tiles - moving_avg(tiles, int(max_period) | 1)
    win = torch.hann_window(T, periodic=False, device=DEV)[:, None]
    spec = torch.fft.rfft(s * win, dim=0).abs() ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0).to(DEV)
    band = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    spec = spec / (spec[band].median(0).values + 1e-12)  # whiten each tile
    P = spec.mean(1)
    Pb = torch.where(band, P, torch.zeros_like(P))
    i = int(Pb.argmax())
    strength = float(Pb[i] / (P[band].median() + 1e-12))
    # parabolic peak refinement
    if 0 < i < len(P) - 1:
        a, b, c = float(P[i - 1]), float(P[i]), float(P[i + 1])
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        off = 0.0
    f0 = (i + max(-0.5, min(0.5, off))) / T
    # if a strong sub-harmonic exists, the true fundamental is lower (PWM/rectified waveforms)
    j = int(round(i / 2))
    if freqs[j] >= 1.0 / max_period and P[j] > 0.35 * P[i]:
        f0 = f0 / 2
    return f0, strength


@torch.no_grad()
def detect_fundamental_mc(logY, ff, valid_min=0.3, min_period=2.0, max_period=40.0):
    """Motion-compensated flicker frequency detection.

    Spectra of the frame-to-frame change of log luminance *along optical flow*, averaged per tile.
    Moving content is followed by the flow and cancels; illumination change does not. The difference
    operator scales a component at f by 2 sin(pi f), which is divided out. Returns (f0, strength).
    """
    T, _, h, w = logY.shape
    gy, gx = 12, 12
    D = torch.zeros(T - 1, gy * gx, device=DEV)
    for t in range(T - 1):
        fl, val = ff.align(t, 1)
        yk, ins = warp(logY[t + 1:t + 2], fl)
        v = (val & ins).float()[:, None]
        d = (yk - logY[t:t + 1]) * v
        num = F.adaptive_avg_pool2d(d, (gy, gx)).reshape(-1)
        den = F.adaptive_avg_pool2d(v, (gy, gx)).reshape(-1)
        D[t] = torch.where(den > valid_min, num / den.clamp_min(1e-6), torch.zeros_like(num))
    n = T - 1
    win = torch.hann_window(n, periodic=False, device=DEV)[:, None]
    spec = torch.fft.rfft((D - D.mean(0)) * win, dim=0).abs() ** 2
    freqs = torch.fft.rfftfreq(n, d=1.0).to(DEV)
    gain = (2 * torch.sin(math.pi * freqs)).clamp_min(0.15) ** 2
    spec = spec / gain[:, None]
    band = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    spec = spec / (spec[band].median(0).values + 1e-12)
    P = spec.mean(1)
    Pb = torch.where(band, P, torch.zeros_like(P))
    i = int(Pb.argmax())
    strength = float(Pb[i] / (P[band].median() + 1e-12))
    if 0 < i < len(P) - 1:
        a, b, c = float(P[i - 1]), float(P[i]), float(P[i + 1])
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        off = 0.0
    f0 = (i + max(-0.5, min(0.5, off))) / n
    j = int(round(i / 2))
    if freqs[j] >= 1.0 / max_period and P[j] > 0.35 * P[i]:
        f0 = f0 / 2
    return f0, strength


def resolvable(comps, K, max_cond=20.0):
    """Greedily keep components (in harmonic order) while the window's design matrix
    [1, k/K, cos, sin, ...] over k = -K..K stays well conditioned. Genuine lamp harmonics
    score 1-2.5 (16 for a 2-component near-Nyquist pair); unresolvable bunches score 100-1000."""
    kk = np.arange(-K, K + 1, dtype=np.float64)
    cols = [np.ones_like(kk), kk / K]
    keep = []
    for fa in comps:
        ang = 2 * math.pi * fa * kk
        A = np.stack(cols + [np.cos(ang), np.sin(ang)], 1)
        A = A / np.linalg.norm(A, axis=0)
        s = np.linalg.svd(A, compute_uv=False)
        if s[0] / s[-1] <= max_cond:
            cols += [np.cos(ang), np.sin(ang)]
            keep.append(fa)
    return keep


@torch.no_grad()
def analyze(frames, clip_name, fps, f0=None, harmonics=8, keep_gain=True, work_edge=WORK_EDGE, log_eps=LOG_EPS,
            flow_tag="", score_frames=None, periodic=None, osc_events=False):
    """Run FlickerScope on frames (T,H,W,3 uint8). Flow is taken from clip_name's cache.

    Pass f0 from the original clip when scoring a processed version so both are measured
    against the same flicker frequency.
    """
    T, H, W, _ = frames.shape
    h, w = work_size(H, W, work_edge)
    Y, clip = lin_luma_stack(frames, h, w)
    logY = torch.log(Y + log_eps)

    ff = FlowField(clip_name, h, w, flow_tag)
    if f0 is None:
        f0, strength = detect_fundamental_mc(logY, ff)
    else:
        strength = float("nan")
    # threshold from validation: genuine flicker scored >= 160 on synthetic and real clips, motion
    # artefacts on raw spectra and a footage clip without clear periodic flicker scored ~12
    auto_periodic = periodic is None
    if periodic is None:  # callers scoring a processed clip pass the original's f0 *and* this decision
        periodic = (strength != strength) or strength > 25.0  # nan => caller fixed f0
    # Harmonics of a fast light (e.g. 100 Hz at 240 fps) alias back below Nyquist, so model each
    # at its aliased frequency: 100 Hz -> 0.417, 200 Hz -> 0.167, 300 Hz -> 0.25 cycles/frame.
    comps = []
    if periodic:
        for k in range(1, harmonics + 1):
            fa = abs(k * f0 - round(k * f0))
            if fa > 0.02 and all(abs(fa - g) > 0.01 for g in comps):
                comps.append(fa)
    period = 1.0 / f0 if f0 > 0 else 12.0
    longest = 1.0 / min(comps) if comps else 12.0
    K = int(min(20, max(4, round(1.5 * longest)))) if periodic else 6
    # a component whose period exceeds the fit window is indistinguishable from the window's trend
    comps = [c for c in comps if 1.0 / c <= 2 * K + 1]
    # Near Nyquist the aliased harmonics bunch up (f0 = 0.495 puts the 3rd, 5th and 7th at 0.485,
    # 0.475, 0.465): a 2K+1 frame window cannot tell such sinusoids apart, the fit turns ill-posed and
    # the coefficients explode. Keep a harmonic only if the window can still resolve it from the rest.
    comps = resolvable(comps, K)
    nh = len(comps)

    PER = torch.zeros(T, h, w, device=DEV)
    AP = torch.zeros(T, h, w, device=DEV)
    sig = torch.zeros(T, h, w, device=DEV)
    coef1 = torch.zeros(T, 2, h, w, device=DEV)  # first harmonic (cos, sin) for phase maps
    TH = torch.zeros(T, 2 * nh, h, w, device=DEV, dtype=torch.float16)  # debiased harmonic coefficients
    NE = torch.zeros(T, h, w, device=DEV)  # valid samples behind each fit
    sb = 1.0  # spatial pooling of the normal equations (work-res pixels)
    sa = max(h, w) / 40.0  # aperiodic smoothness scale

    for t in range(T):
        nb = ff.neighbors(t, K)
        ks = sorted(nb)
        Yk, Vk = [], []
        for k in ks:
            fl, val = nb[k]
            yk, ins = warp(logY[t + k:t + k + 1], fl)
            ck, _ = warp(clip[t + k:t + k + 1].float(), fl)
            Yk.append(yk[0, 0])
            Vk.append((val[0] & ins[0] & (ck[0, 0] < 0.5)).float())
        Yk = torch.stack(Yk)
        Vk = torch.stack(Vk)
        kk = torch.tensor(ks, device=DEV, dtype=torch.float32)
        cols = [torch.ones_like(kk), kk / K]
        for fa in comps:
            ang = 2 * math.pi * fa * (t + kk)
            cols += [torch.cos(ang), torch.sin(ang)]
        A = torch.stack(cols, 1)  # (n, p)
        p = A.shape[1]
        Wk = Vk.clone()
        for it in range(5):
            M = torch.einsum("nhw,ni,nj->ijhw", Wk, A, A).reshape(1, p * p, h, w)
            r = torch.einsum("nhw,ni,nhw->ihw", Wk, A, Yk).reshape(1, p, h, w)
            M = gauss_blur(M, sb).reshape(p, p, h * w).permute(2, 0, 1)
            r = gauss_blur(r, sb).reshape(p, h * w).T
            lam = 1e-4 * M.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-6)
            M = M + lam[:, None, None] * torch.eye(p, device=DEV)
            th = torch.linalg.solve(M, r[..., None])[..., 0].T.reshape(p, h, w)
            pred = torch.einsum("ni,ihw->nhw", A, th)
            e = Yk - pred
            s2 = (Wk * e ** 2).sum(0) / Wk.sum(0).clamp_min(1.0)  # scale from inliers, not from the flash
            s = s2.sqrt().clamp_min(1e-3)
            if it < 2:   # Huber first (convex, stable start) ...
                Wk = Vk * torch.clamp(1.5 * s / (e.abs() + 1e-9), max=1.0)
            else:        # ... then Tukey biweight: gross outliers (a muzzle flash) get exactly zero weight
                u = e / (4.685 * s)
                Wk = Vk * torch.clamp(1 - u ** 2, min=0) ** 2
        n_eff = Wk.sum(0).clamp_min(1.0)

        # periodic part at k = 0, harmonics debiased against the noise floor
        per = torch.zeros(h, w, device=DEV)
        cd = []
        for hh in range(nh):
            c, d = th[2 + 2 * hh], th[3 + 2 * hh]
            Ph = c ** 2 + d ** 2
            noise = 4.0 * 4.0 * s ** 2 / n_eff
            g = torch.sqrt(torch.clamp((Ph - noise) / (Ph + 1e-12), 0.0, 1.0))
            cd.append((c * g, d * g))
        if nh:  # backstop: no lamp swings beyond e^+-4 (genuine fits peak near 3.2 on the range clip)
            amp = torch.sqrt(sum(c ** 2 + d ** 2 for c, d in cd))
            lim = torch.clamp(MAX_AMP / (amp + 1e-12), max=1.0)
            cd = [(c * lim, d * lim) for c, d in cd]
        for hh, (c, d) in enumerate(cd):
            ang = 2 * math.pi * comps[hh] * t
            per = per + c * math.cos(ang) + d * math.sin(ang)
            TH[t, 2 * hh], TH[t, 2 * hh + 1] = c.half(), d.half()

        # aperiodic: spatially smooth part of the frame-t residual (normalised convolution)
        i0 = ks.index(0)
        res = Yk[i0] - th[0] - per
        m = Vk[i0]
        num = gauss_blur((res * m)[None, None], sa)[0, 0]
        den = gauss_blur(m[None, None], sa)[0, 0].clamp_min(1e-3)
        ap = num / den
        floor = 2.0 * s / (2.0 * math.sqrt(math.pi) * sa)
        ap = torch.sign(ap) * torch.clamp(ap.abs() - floor, min=0.0)
        ap = ap * m  # never extrapolate an aperiodic correction into pixels clipped/invalid in this frame

        PER[t] = per
        AP[t] = ap
        sig[t] = s
        NE[t] = Vk.sum(0)

    # ---------------------------------------------------------------- support check
    # Where flow cannot track the content (rain, spray, confetti) few samples align, and a fit with
    # more harmonic coefficients than samples interpolates them exactly: zero residual, no noise
    # floor, coefficients of +-50 in log units. If most fits are underdetermined, the clip cannot
    # carry a per-pixel periodic model; the frame-level (aperiodic) model needs only the trend.
    starved = float((NE < 2 + 2 * nh).float().mean()) if nh else 0.0
    if nh and auto_periodic and starved > 0.5:
        del PER, AP, sig, coef1, TH, NE
        out = analyze(frames, clip_name, fps, f0=f0, harmonics=harmonics, keep_gain=keep_gain, work_edge=work_edge,
                      log_eps=log_eps, flow_tag=flow_tag, score_frames=score_frames, periodic=False,
                      osc_events=osc_events)
        out.update(strength=strength, fallback=f"periodic model unsupported: {starved:.0%} of fits underdetermined")
        return out

    # ---------------------------------------------------------------- starved frames
    # Where a frame's fit has too few valid samples (flow into and out of a flash frame fails its
    # consistency check, so almost nothing aligns), the harmonic coefficients are unconstrained.
    # Lamp flicker is stable over time, so such frames borrow coefficients from nearby frames,
    # weighted by their support. Well-supported frames keep their own fit (no smearing at moving edges).
    if nh:
        Ks = max(2, K // 2)
        rho = torch.clamp((NE - 4.0) / 8.0, 0, 1)
        for t in range(T):
            if bool((rho[t] > 0.999).all()):
                coef1[t] = TH[t, 0:2].float()
                continue
            num = torch.zeros(2 * nh, h, w, device=DEV)
            den = torch.zeros(h, w, device=DEV)
            for tau in range(max(0, t - Ks), min(T, t + Ks + 1)):
                if tau == t:
                    continue
                wt = NE[tau] * rho[tau] * (1 - abs(tau - t) / (Ks + 1))
                num += TH[tau].float() * wt
                den += wt
            nbth = num / den.clamp_min(1e-6)
            r_t = torch.where(den > 1e-6, rho[t], torch.ones_like(rho[t]))
            th_t = TH[t].float() * r_t + nbth * (1 - r_t)
            per = torch.zeros(h, w, device=DEV)
            for hh in range(nh):
                ang = 2 * math.pi * comps[hh] * t
                per = per + th_t[2 * hh] * math.cos(ang) + th_t[2 * hh + 1] * math.sin(ang)
            PER[t] = per
            AP[t] = AP[t] * r_t  # the aperiodic residual of an unconstrained fit is not trustworthy either
            coef1[t] = th_t[0:2]
            TH[t] = th_t.half()

    # ---------------------------------------------------------------- events
    # A muzzle flash or a lamp switching on is a one-off transient, not flicker. Aperiodic
    # deviations far outside a pixel's usual fluctuation are classified as events and left alone.
    med = AP.median(0).values
    mad = (AP - med).abs().median(0).values * 1.4826
    thr = 3.5 * mad.clamp_min(0.02)
    dev = (AP - med).abs()
    ev = dev > thr
    near = dev > 0.5 * thr
    # +28% beyond what the flicker model predicts is an unmistakable transient (a muzzle flash is
    # +250%): always an event, whatever else happens at that pixel
    strong = dev > torch.maximum(thr, torch.full_like(thr, 0.25))
    ev = ev | (near & (F.max_pool1d(ev.float().permute(1, 2, 0).reshape(-1, 1, T), 3, 1, 1)
                       .reshape(h, w, T).permute(2, 0, 1) > 0))
    # events are spatially coherent: grow strong cores into adjacent moderately-deviating pixels
    for _ in range(3):
        grown = F.max_pool2d(ev.float()[:, None], 5, stride=1, padding=2)[:, 0] > 0
        ev = ev | (grown & near)
    # Recurring "events" are model misfit on flicker, not events. Count separate episodes (rising edges),
    # not frames: a shot followed by lingering smoke is one long episode and must stay an event.
    starts = (ev[1:] & ~ev[:-1]).sum(0) + ev[0].long()
    recurring = (starts > max(3, int(0.03 * T)))[None]
    # strong cores grow into moderately deviating surroundings, and into pixels clipped in that frame
    # (a flash's saturated centre has no usable residual of its own)
    reach = near | clip[:, 0]
    for _ in range(4):
        strong = strong | ((F.max_pool2d(strong.float()[:, None], 5, stride=1, padding=2)[:, 0] > 0) & reach)
    ev = (ev & ~recurring) | strong
    if osc_events:
        # Light that swings the other way both before and after (brighter, darker, brighter) is flicker
        # the model missed, not an event. A flash or a lamp switching on is one-sided: calm before it.
        D = AP - med
        big = D.abs() > thr

        def opposite(shifts):
            o = torch.zeros_like(ev)
            for k in shifts:
                Dk = torch.roll(D, k, 0)
                okk = (Dk * D < 0) & (Dk.abs() > 0.5 * D.abs()) & torch.roll(big, k, 0)
                if k > 0:
                    okk[:k] = False
                else:
                    okk[k:] = False
                o = o | okk
            return o

        osc = opposite((1, 2, 3)) & opposite((-1, -2, -3))
        ev = ev & ~osc
    # A flash lights the whole frame (faintly far from the source), so when much of a frame is an
    # event the entire frame's aperiodic part is the event: its gain is the periodic lamp model only.
    frame_ev = ev.float().mean((1, 2)) > 0.25
    if DEBUG is not None:
        DEBUG.update(AP_raw=AP.clone(), AP_med=med.clone(), ev_pixel=ev.clone(), frame_ev=frame_ev.clone())
    ev = ev | frame_ev[:, None, None]
    AP = torch.where(ev, torch.zeros_like(AP), AP)

    logG = PER + AP

    # ---------------------------------------------------------------- chronically clipped pixels
    # A pixel that saturates in every bright phase never shows the fit how bright those phases are,
    # so its gain is underestimated. The lamp lighting it also lights its unclipped surroundings and
    # the gain field is smooth across a surface: fill such pixels from their unclipped neighbourhood.
    # Clipping is judged in a local time window, not over the whole clip: the camera may move, and
    # a position that showed a blown-out wall earlier may show something else now.
    clipfrac = moving_avg(clip.float()[:, 0].reshape(T, -1), 2 * K + 1).reshape(T, h, w)
    good = (clipfrac < 0.05).float()
    if bool((clipfrac > 0.1).any()):
        sf = max(2.0, max(h, w) / 80.0)
        num = gauss_blur((logG * good)[:, None], sf)[:, 0]
        den = gauss_blur(good[:, None], sf)[:, 0]
        fill = num / den.clamp_min(1e-3)
        a = torch.clamp((clipfrac - 0.05) / 0.15, 0, 1) * (den > 0.05).float()
        logG = logG * (1 - a) + fill * a

    # ---------------------------------------------------------------- arithmetic-mean normalisation
    # The Fourier fit gives log G zero mean, i.e. G normalised to its *geometric* mean. Perceived light
    # (and a long exposure) is the *arithmetic* mean, which for deep flicker is far higher (at 87%
    # modulation frames differ ~14x and the geometric mean is ~38% below the arithmetic one). Without
    # this, the deeper a region flickers the darker its corrected steady level: uneven darkening.
    if nh:
        P = None
        for cand in range(2, 31):
            if all(abs(c * cand - round(c * cand)) < 0.03 for c in comps):
                P = cand
                break
        N = P or 60
        logm = torch.zeros(T, h, w, device=DEV)
        fa = torch.tensor(comps, device=DEV, dtype=torch.float32)[:, None]   # (nh, 1)
        nn = torch.arange(N, device=DEV, dtype=torch.float32)[None, :]      # (1, N)
        for t in range(T):
            # the lamp waveform at every phase of one cycle, all at once: rows interleave cos/sin
            # in TH's order (c_j, d_j), so e[n] = sum_j c_j cos(2 pi f_j (t+n)) + d_j sin(...)
            ang = 2 * math.pi * fa * (t + nn)
            basis = torch.stack([torch.cos(ang), torch.sin(ang)], 1).reshape(2 * nh, N)
            e = torch.einsum("kn,khw->nhw", basis, TH[t].float())
            logm[t] = torch.log(torch.exp(e).mean(0))
        # the normalisation is a property of the lamp: keep it stable over time, or coefficient noise,
        # exponentiated, turns into frame-to-frame brightness wobble in deep-flicker regions
        logm = moving_avg(logm.reshape(T, -1), 2 * K + 1).reshape(T, h, w)
        logG = logG - logm

    # ---------------------------------------------------------------- scores
    G = torch.exp(logG)
    Gs = G if score_frames is None else G[torch.as_tensor(score_frames, device=DEV)]
    q_lo = torch.quantile(Gs.reshape(len(Gs), -1), 0.02, dim=0).reshape(h, w)
    q_hi = torch.quantile(Gs.reshape(len(Gs), -1), 0.98, dim=0).reshape(h, w)
    pct = (q_hi - q_lo) / (q_hi + q_lo)
    Gm = Gs.mean(0)
    fidx = torch.clamp(Gs - Gm, min=0).sum(0) / Gs.sum(0)
    vis = torch.clamp(Y.mean(0)[0] / 0.01, 0, 1) * (1.0 - clip.float().mean(0)[0])
    wsum = vis.sum().clamp_min(1e-6)
    pv = pct.flatten()
    order = torch.argsort(pv)
    cw = torch.cumsum(vis.flatten()[order], 0) / wsum
    peak = float(pv[order][torch.searchsorted(cw, torch.tensor([0.95], device=DEV)).clamp(max=len(pv) - 1)])
    frame_curve = (logG * vis).sum((1, 2)) / wsum
    ev_area = (ev.float() * vis).sum((1, 2)) / wsum
    events = [int(i) for i in torch.nonzero(ev_area > 0.05).flatten()]

    phase = torch.atan2(coef1[:, 1], coef1[:, 0])
    amp1 = coef1.norm(dim=1)
    phase_map = torch.atan2((amp1 * torch.sin(phase)).mean(0), (amp1 * torch.cos(phase)).mean(0))

    out = dict(
        flicker_pct=float((pct * vis).sum() / wsum * 100),
        coverage_pct=float((vis * (pct > PERCEPTIBLE)).sum() / wsum * 100),
        peak_pct=peak * 100,
        flicker_index=float((fidx * vis).sum() / wsum),
        f0=f0, hz=f0 * fps, period_frames=period, strength=strength, periodic=bool(periodic),
        harmonics=nh, components=comps, window=K, work_size=[h, w], events=events,
        noise_sigma=float((sig.mean(0) * vis).sum() / wsum),
        maps=dict(pct=pct.cpu().numpy(), vis=vis.cpu().numpy(), phase=phase_map.cpu().numpy(),
                  frame_curve=frame_curve.cpu().numpy()),
    )
    if keep_gain:
        out["logG"] = logG  # (T,h,w) on DEV
        out["event_mask"] = ev
    if DEBUG is not None:
        DEBUG.update(PER=PER, AP=AP, NE=NE, sig=sig, TH=TH, logm=logm if nh else None)
    return out


DEBUG = None  # set to a dict to receive the fit's intermediate fields


def summary(r):
    return (f"Flicker {r['flicker_pct']:5.1f}%  coverage {r['coverage_pct']:5.1f}%  peak {r['peak_pct']:5.1f}%  "
            f"FI {r['flicker_index']:.3f}  f0 {r['f0']:.4f} c/f ({r['hz']:.2f} Hz, period {r['period_frames']:.2f}f, "
            f"strength {r['strength']:.1f}, {'periodic' if r['periodic'] else 'aperiodic'})")


if __name__ == "__main__":
    import argparse
    import time
    from common import list_clips, load_frames, load_meta
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    a = ap.parse_args()
    for name in a.clips or list_clips():
        meta = load_meta(name)
        t0 = time.time()
        r = analyze(load_frames(name), name, meta["fps"], keep_gain=False)
        print(f"{name:24s} {summary(r)}  [{time.time() - t0:.1f}s]")
