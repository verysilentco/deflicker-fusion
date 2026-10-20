"""Render site/index.html from site/template.html + data-driven SVG figures (CPU only, no data needed
beyond site/*.json). The rendered page plus site/media/ is what the Hugging Face Space serves.

python make_page.py                 -> site/index.html
python make_page.py path/to/space   -> path/to/space/index.html
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(os.path.dirname(HERE), "site")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------ figure 1: pipeline (hand-laid grid)
def fig_pipeline():
    W, H = 1100, 500
    bw, bh = 164, 74
    xs = [212, 404, 596, 788]
    yA, yB = 92, 334
    top = [("Optical flow", "RAFT on frames with the", "slow light divided out"),
           ("Flicker frequency", "spectrum of change along", "the flow → f0, aliased set"),
           ("Per-pixel fit", "trend + Fourier series", "along trajectories, Tukey"),
           ("Events · normalise", "flash = event, not flicker;", "gain to arithmetic mean")]
    bot = [("Align & divide", "warp frames t±k onto t,", "÷ each one's own gain"),
           ("SNR-weighted merge", "weight ∝ gain²; reject", "clipped & misaligned"),
           ("Event light", "add the flash back; flash", "frames rebuilt from t−P"),
           ("Refine · guard", "full-res gain, 2nd pass;", "never add flicker")]
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="FlickerScope measures the per-pixel flicker gain, '
         f'events and light period; Flicker Fusion uses them to rebuild each frame from flow-aligned neighbours." '
         f'class="fig fig-wide">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>'
         '<marker id="ahL" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="f-lamp"/></marker>'
         '<marker id="ahS" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="f-steady"/></marker></defs>']
    # lane labels
    o.append(f'<text x="{xs[0]}" y="{yA - 22}" class="t-lane s-lamp">FLICKERSCOPE · MEASURE</text>')
    o.append(f'<text x="{xs[0]}" y="{yB - 22}" class="t-lane s-steady">FLICKER FUSION · REMOVE</text>')
    # input
    ix, iy, iw, ih = 20, 196, 150, 104
    for k in range(3):
        o.append(f'<rect x="{ix + 16 - 8 * k}" y="{iy + 8 * k - 8}" width="{iw - 24}" height="{ih - 22}" rx="6" '
                 f'class="box-soft"/>')
    o.append(f'<text x="{ix + iw / 2 - 4}" y="{iy + 40}" class="t-box" text-anchor="middle">Input video</text>')
    o.append(f'<text x="{ix + iw / 2 - 4}" y="{iy + 58}" class="t-sub" text-anchor="middle">I(x, t), slow-mo</text>')
    # boxes
    def box(x, y, title, a, b, cls):
        return (f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" class="box {cls}"/>'
                f'<text x="{x + 12}" y="{y + 24}" class="t-box">{esc(title)}</text>'
                f'<text x="{x + 12}" y="{y + 44}" class="t-sub">{esc(a)}</text>'
                f'<text x="{x + 12}" y="{y + 60}" class="t-sub">{esc(b)}</text>')
    for x, (t, a, b) in zip(xs, top):
        o.append(box(x, yA, t, a, b, "box-lamp"))
    for x, (t, a, b) in zip(xs, bot):
        o.append(box(x, yB, t, a, b, "box-steady"))
    # input -> lanes
    o.append(f'<path d="M{ix + iw - 8} {iy + 24} C 190 {iy}, 190 {yA + bh / 2}, {xs[0] - 4} {yA + bh / 2}" '
             f'class="edge" marker-end="url(#ah)"/>')
    o.append(f'<path d="M{ix + iw - 8} {iy + 70} C 190 {iy + 100}, 190 {yB + bh / 2}, {xs[0] - 4} {yB + bh / 2}" '
             f'class="edge" marker-end="url(#ah)"/>')
    # lane arrows with labels
    labA = ["flow", "f0, harmonics", "fit residual"]
    labB = ["S estimates", "merged S", "with events"]
    for i in range(3):
        x1, x2 = xs[i] + bw, xs[i + 1] - 4
        # in-lane arrows mean "next step"; the labelled hand-offs between lanes carry the mechanism
        o.append(f'<line x1="{x1}" y1="{yA + bh / 2}" x2="{x2}" y2="{yA + bh / 2}" class="edge e-lamp" marker-end="url(#ahL)"/>')
        o.append(f'<line x1="{x1}" y1="{yB + bh / 2}" x2="{x2}" y2="{yB + bh / 2}" class="edge e-steady" marker-end="url(#ahS)"/>')
    # cross-lane hand-offs (what the meter gives the remover)
    def drop(xa, xb, label, dy=0):
        ya, yb = yA + bh, yB - 4
        mid = (ya + yb) / 2 + dy
        return (f'<path d="M{xa} {ya} L{xa} {mid} L{xb} {mid} L{xb} {yb}" class="edge e-hand" marker-end="url(#ah)"/>'
                f'<text x="{(xa + xb) / 2 if xa != xb else xa + 8}" y="{mid - 7}" class="t-hand" '
                f'text-anchor="{"middle" if xa != xb else "start"}">{label}</text>')
    o.append(drop(xs[0] + 40, xs[0] + 40, "flow field", -10))
    o.append(drop(xs[3] + 60, xs[0] + 120, "gain G(x, t) per pixel, per frame", 14))
    o.append(drop(xs[3] + 120, xs[2] + 120, "event mask · period P", -14))
    # outputs
    ox = 978
    o.append(f'<line x1="{xs[3] + bw}" y1="{yA + bh / 2}" x2="{ox - 4}" y2="{yA + bh / 2}" class="edge e-lamp" marker-end="url(#ahL)"/>')
    o.append(f'<rect x="{ox}" y="{yA + 6}" width="104" height="{bh - 12}" rx="31" class="pill pill-lamp"/>')
    o.append(f'<text x="{ox + 52}" y="{yA + 33}" class="t-box" text-anchor="middle">Flicker %</text>')
    o.append(f'<text x="{ox + 52}" y="{yA + 50}" class="t-sub" text-anchor="middle">+ per-pixel maps</text>')
    o.append(f'<line x1="{xs[3] + bw}" y1="{yB + bh / 2}" x2="{ox - 4}" y2="{yB + bh / 2}" class="edge e-steady" marker-end="url(#ahS)"/>')
    o.append(f'<rect x="{ox}" y="{yB + 6}" width="104" height="{bh - 12}" rx="31" class="pill pill-steady"/>')
    o.append(f'<text x="{ox + 52}" y="{yB + 33}" class="t-box" text-anchor="middle">Steady video</text>')
    o.append(f'<text x="{ox + 52}" y="{yB + 50}" class="t-sub" text-anchor="middle">same codec, audio</text>')
    o.append('</svg>')
    return "".join(o)


# ------------------------------------------------------------------ figure 2: aliasing (exact samples)
def fig_aliasing():
    a = json.load(open(os.path.join(SITE, "aliasing.json")))
    W, H = 760, 300
    x0, x1 = 56, 730
    X = lambda u: x0 + u * (x1 - x0)
    yT0, yT1 = 112, 34  # light 0 -> 1
    Yt = lambda v: yT0 + v * (yT1 - yT0)
    yB0, yB1 = 262, 172
    pts = " ".join(f"{X(u):.1f},{Yt(v):.1f}" for u, v in zip(a["t"], a["light"]))
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="A 100 Hz lamp sampled at 240 frames per second: '
         f'each frame catches a different point of the light cycle, and the pattern repeats every 12 frames." class="fig">']
    o.append(f'<text x="{x0}" y="22" class="t-sub">lamp output, 50 Hz mains → 100 Hz</text>')
    o.append(f'<line x1="{x0}" y1="{yT0}" x2="{x1}" y2="{yT0}" class="axis"/>')
    o.append(f'<polyline points="{pts}" class="wave"/>')
    for i, (u, v) in enumerate(zip(a["st"], a["samples"])):
        x = X(u)
        o.append(f'<line x1="{x:.1f}" y1="{Yt(v):.1f}" x2="{x:.1f}" y2="{yB1 - 18}" class="tick"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{Yt(v):.1f}" r="4.2" class="dot"/>')
        bh = v * (yB0 - yB1)
        o.append(f'<rect x="{x - 15:.1f}" y="{yB0 - bh:.1f}" width="30" height="{max(bh, 1.5):.1f}" rx="3" class="bar"/>')
        o.append(f'<text x="{x:.1f}" y="{yB0 + 18}" class="t-tiny" text-anchor="middle">{i + 1}</text>')
    o.append(f'<line x1="{x0}" y1="{yB0}" x2="{x1}" y2="{yB0}" class="axis"/>')
    o.append(f'<text x="{x0}" y="{yB1 - 26}" class="t-sub">brightness of each 1/240 s frame</text>')
    o.append(f'<text x="{x1}" y="{yB1 - 26}" class="t-sub" text-anchor="end">frames 1–12 = 50 ms = 5 lamp cycles</text>')
    o.append('</svg>')
    return "".join(o)


# ------------------------------------------------------------------ figure 3: fusion around a dark frame
def fig_fusion():
    c = json.load(open(os.path.join(SITE, "curves.json")))
    orig = np.array(c["original"])
    t = 13  # a darkest-phase frame in the range clip
    ks = list(range(-3, 4))
    g = np.array([1 + orig[t + k] / 100 for k in ks])  # light level of each frame vs steady
    wts = g ** 2 / (g ** 2).sum()
    W, H = 760, 300
    cx = [80 + i * 88 for i in range(len(ks))]
    yb, hmax = 150, 100
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Frame {t + 1} caught the lamps at their dimmest; '
         f'its neighbours were lit up to {g.max():.2f}x. Each neighbour is aligned and divided by its own light level, '
         f'then weighted by light level squared, so the dark frame is rebuilt mostly from bright neighbours." class="fig">']
    o.append(f'<text x="40" y="22" class="t-sub">light level of each frame (range clip, real values)</text>')
    o.append(f'<line x1="40" y1="{yb}" x2="{cx[-1] + 40}" y2="{yb}" class="axis"/>')
    ref = yb - hmax / g.max()
    o.append(f'<line x1="40" y1="{ref:.1f}" x2="{cx[-1] + 40}" y2="{ref:.1f}" class="axis dash"/>')
    o.append(f'<text x="{cx[-1] + 44}" y="{ref + 4:.1f}" class="t-tiny">steady</text>')
    out_x, out_y = 690, 248
    for i, k in enumerate(ks):
        h = hmax * g[i] / g.max()
        cls = "bar-target" if k == 0 else "bar"
        o.append(f'<rect x="{cx[i] - 22}" y="{yb - h:.1f}" width="44" height="{h:.1f}" rx="4" class="{cls}"/>')
        o.append(f'<text x="{cx[i]}" y="{yb - h - 7:.1f}" class="t-tiny" text-anchor="middle">{g[i]:.2f}×</text>')
        o.append(f'<text x="{cx[i]}" y="{yb + 17}" class="t-tiny" text-anchor="middle">{"t" if k == 0 else f"t{k:+d}"}</text>')
        sw = 1 + 9 * wts[i] / wts.max()
        o.append(f'<path d="M{cx[i]} {yb + 26} C {cx[i]} {out_y - 10}, {out_x - 120} {out_y}, {out_x - 6} {out_y}" '
                 f'class="flow" style="stroke-width:{sw:.2f}"/>')
        o.append(f'<text x="{cx[i]}" y="{yb + 40}" class="t-tiny t-w" text-anchor="middle">{wts[i] * 100:.0f}%</text>')
    o.append(f'<rect x="{out_x}" y="{out_y - 24}" width="58" height="48" rx="8" class="box box-steady"/>')
    o.append(f'<text x="{out_x + 29}" y="{out_y + 5}" class="t-box" text-anchor="middle">t</text>')
    o.append(f'<text x="40" y="{H - 8}" class="t-sub">weight ∝ light² — line width and % show each frame’s share</text>')
    o.append('</svg>')
    return "".join(o)


# ------------------------------------------------------------------ figure 4: brightness curve, before/after
def fig_curve():
    c = json.load(open(os.path.join(SITE, "curves.json")))
    a, b = np.array(c["original"]), np.array(c["fusion"])
    T = len(a)
    W, H = 900, 250
    x0, x1, y0, y1 = 52, 884, 222, 22
    lo, hi = -40, 30
    X = lambda i: x0 + i / (T - 1) * (x1 - x0)
    Y = lambda v: y0 + (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Frame brightness relative to the steady level: the '
         f'original swings between about {a.min():.0f}% and +{a.max():.0f}% every few frames; Flicker Fusion stays '
         f'within a few percent, except the muzzle flash, which is kept." class="fig fig-wide">']
    for v in (-40, -20, 0, 20):
        o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" class="grid{" zero" if v == 0 else ""}"/>')
        o.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" class="t-tiny" text-anchor="end">{v:+d}%</text>')
    for f in range(0, T, 25):
        o.append(f'<text x="{X(f):.1f}" y="{y0 + 18}" class="t-tiny" text-anchor="middle">{f}</text>')
    ev = [e for e in c["events"] if 150 <= e <= 160]
    if ev:
        o.append(f'<rect x="{X(min(ev)) - 3:.1f}" y="{y1}" width="{X(max(ev)) - X(min(ev)) + 6:.1f}" height="{y0 - y1}" class="evband"/>')
        o.append(f'<text x="{X(157):.1f}" y="{y1 + 12}" class="t-tiny" text-anchor="middle">shot</text>')
    o.append(f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(a))}" class="line-lamp"/>')
    o.append(f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(b))}" class="line-steady"/>')
    o.append('</svg>')
    return "".join(o)


def limits_rows():
    p = os.path.join(SITE, "limits.json")
    rows = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    return "".join(f'<tr><td><b>{esc(r["title"])}</b><br><span class="k">{esc(r["note"])}</span></td>'
                   f'<td class="num">{r["before"]:.1f} → {r["after"]:.1f}</td><td>{esc(r["why"])}</td></tr>' for r in rows)


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else SITE
    tpl = open(os.path.join(SITE, "template.html"), encoding="utf-8").read()
    exj = os.path.join(SITE, "examples.json")
    ex = open(exj, encoding="utf-8").read() if os.path.exists(exj) else "[]"
    js = open(os.path.join(SITE, "player.js"), encoding="utf-8").read()
    page = (tpl.replace("{{FIG_PIPELINE}}", fig_pipeline()).replace("{{FIG_ALIASING}}", fig_aliasing())
            .replace("{{FIG_FUSION}}", fig_fusion()).replace("{{FIG_CURVE}}", fig_curve())
            .replace("{{EXAMPLES_JSON}}", ex).replace("{{PLAYER_JS}}", js).replace("{{LIMITS_ROWS}}", limits_rows()))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print("wrote", os.path.join(out_dir, "index.html"), round(len(page) / 1024), "KB")
