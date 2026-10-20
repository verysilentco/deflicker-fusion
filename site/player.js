(function () {
  "use strict";
  var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var pool = document.getElementById("pool");

  /** A wipe player: one video holding original | processed side by side, drawn on a canvas, so the
      two halves can never drift out of sync. Videos stay loaded, so switching keeps the playhead. */
  function makePlayer(root) {
    var stage = root.querySelector(".stage"), cv = stage.querySelector("canvas"), ctx = cv.getContext("2d");
    var hint = stage.querySelector(".hint"), chipR = stage.querySelector(".chipR"), frameEl = root.querySelector(".frame");
    var playBtn = root.querySelector(".play"), rates = Array.prototype.slice.call(root.querySelectorAll(".rate"));
    var fps = +root.dataset.fps || 30, frames = +root.dataset.frames || 100;
    var vids = {}, active = null, wipe = 0.5, target = null, vel = 0, dragging = false, grab = 0, W = 1, H = 1, DPR = 1;
    var pressed = root.querySelector('.rate[aria-pressed="true"]');
    var rate = pressed ? +pressed.dataset.rate : 1;

    function video(src) {
      if (vids[src]) return vids[src];
      var v = document.createElement("video");
      v.src = src; v.muted = true; v.loop = true; v.playsInline = true; v.preload = "auto";
      v.setAttribute("playsinline", ""); v.setAttribute("muted", "");
      v.style.width = "2px"; v.style.height = "2px"; pool.appendChild(v); vids[src] = v;
      v.addEventListener("loadedmetadata", function () { if (v === active) shape(); });
      return v;
    }
    function shape() {
      if (!active || !active.videoWidth) return;
      stage.style.aspectRatio = (active.videoWidth / 2) + " / " + active.videoHeight;
      resize();
    }
    function load(src, label, opts) {
      opts = opts || {};
      if (opts.fps) fps = opts.fps;
      if (opts.frames) frames = opts.frames;
      if (label) chipR.textContent = label;
      var next = video(src), keep = opts.keepTime !== false;
      var t = active && keep ? active.currentTime : 0, playing = active ? !active.paused : true;
      if (active && active !== next) active.pause();
      active = next; active.playbackRate = rate; shape();
      var go = function () {
        try { active.currentTime = keep ? Math.min(t, active.duration || t) : 0; } catch (e) {}
        if (playing) active.play().catch(function () {});
      };
      if (active.readyState >= 1) go(); else active.addEventListener("loadedmetadata", go, { once: true });
      syncPlay();
    }
    function syncPlay() { playBtn.textContent = active && !active.paused ? "Pause" : "Play"; }
    playBtn.addEventListener("click", function () {
      if (!active) return;
      if (active.paused) active.play().catch(function () {}); else active.pause();
      setTimeout(syncPlay, 30);
    });
    function step(d) {
      if (!active) return;
      active.pause(); syncPlay();
      var n = Math.max(0, Math.min(frames - 1, Math.floor(active.currentTime * fps) + d));
      active.currentTime = (n + 0.5) / fps;
    }
    root.querySelector(".prev").addEventListener("click", function () { step(-1); });
    root.querySelector(".next").addEventListener("click", function () { step(1); });
    rates.forEach(function (b) {
      b.addEventListener("click", function () {
        rate = +b.dataset.rate; if (active) active.playbackRate = rate;
        rates.forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
      });
    });
    function resize() {
      var r = stage.getBoundingClientRect(); DPR = window.devicePixelRatio || 1; W = r.width; H = r.height;
      cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    }
    if (window.ResizeObserver) new ResizeObserver(resize).observe(stage); else window.addEventListener("resize", resize);
    resize();
    function fit() {
      var v = active; if (!v || !v.videoWidth) return null;
      var sw = v.videoWidth / 2, sh = v.videoHeight, s = Math.min(W / sw, H / sh);
      return { sw: sw, sh: sh, x: (W - sw * s) / 2, y: (H - sh * s) / 2, w: sw * s, h: sh * s };
    }
    var last = performance.now();
    function tick(now) {
      var dt = Math.min(1 / 30, (now - last) / 1000); last = now;
      if (target !== null) {            // critically damped spring (response 0.4 s), interruptible
        var k = 246.74, c = 31.42, n = 4, h = dt / n;
        for (var i = 0; i < n; i++) { var a = -k * (wipe - target) - c * vel; vel += a * h; wipe += vel * h; }
        if (Math.abs(wipe - target) < 0.001 && Math.abs(vel) < 0.01) { wipe = target; vel = 0; target = null; }
      }
      draw(); requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    function draw() {
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
      var v = active; if (!v || v.readyState < 2) return;
      var f = fit(); if (!f) return;
      var x = f.x + wipe * f.w;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, H); ctx.clip(); ctx.drawImage(v, 0, 0, f.sw, f.sh, f.x, f.y, f.w, f.h); ctx.restore();
      ctx.save(); ctx.beginPath(); ctx.rect(x, 0, W - x, H); ctx.clip(); ctx.drawImage(v, f.sw, 0, f.sw, f.sh, f.x, f.y, f.w, f.h); ctx.restore();
      ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fillRect(x - 2, 0, 4, H); ctx.fillStyle = "#fff"; ctx.fillRect(x - 1, 0, 2, H);
      var ky = H / 2; ctx.beginPath(); ctx.arc(x, ky, 17, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(x - 5, ky - 5); ctx.lineTo(x - 9, ky); ctx.lineTo(x - 5, ky + 5);
      ctx.moveTo(x + 5, ky - 5); ctx.lineTo(x + 9, ky); ctx.lineTo(x + 5, ky + 5); ctx.stroke();
      var fr = Math.min(frames - 1, Math.floor(v.currentTime * fps));
      frameEl.textContent = "frame " + (fr + 1) + " / " + frames;
    }
    function px(e) { var r = stage.getBoundingClientRect(); return e.clientX - r.left; }
    stage.addEventListener("pointerdown", function (e) {
      var f = fit(); if (!f) return;
      stage.setPointerCapture(e.pointerId); dragging = true; target = null; vel = 0; hint.hidden = true;
      var lineX = f.x + wipe * f.w, x = px(e);
      grab = Math.abs(x - lineX) < 28 ? x - lineX : 0;        // keep the grab offset when the handle is taken
      var to = Math.max(0, Math.min(1, (x - f.x) / f.w));
      if (!grab) { if (reduce) wipe = to; else target = to; }
    });
    stage.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      var f = fit(); if (!f) return;
      var want = Math.max(0, Math.min(1, (px(e) - grab - f.x) / f.w));
      if (target === null) wipe = want;
      else if (Math.abs(target - wipe) < 0.02) { target = null; wipe = want; }
      else target = want;
    });
    function end() { dragging = false; }
    stage.addEventListener("pointerup", end); stage.addEventListener("pointercancel", end);
    stage.tabIndex = 0;
    stage.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") wipe = Math.max(0, wipe - .05);
      if (e.key === "ArrowRight") wipe = Math.min(1, wipe + .05);
    });
    document.addEventListener("play", syncPlay, true); document.addEventListener("pause", syncPlay, true);
    return { load: load };
  }

  // hero: one clip, several methods
  var heroRoot = document.getElementById("hero"), hero = makePlayer(heroRoot);
  var mbuttons = Array.prototype.slice.call(heroRoot.querySelectorAll(".methods button"));
  mbuttons.forEach(function (b) {
    b.addEventListener("click", function () {
      mbuttons.forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
      hero.load(b.dataset.src, b.dataset.label);
    });
  });
  hero.load(mbuttons[0].dataset.src, mbuttons[0].dataset.label);

  // examples: several clips, one method
  var EXAMPLES = window.__EXAMPLES__ || [];
  var exRoot = document.getElementById("examples"), cards = document.getElementById("exCards"), cap = document.getElementById("exCaption");
  if (!EXAMPLES.length) { document.getElementById("more").hidden = true; return; }
  var ex = makePlayer(exRoot), buttons = [];
  function pick(i) {
    var e = EXAMPLES[i];
    buttons.forEach(function (b, j) { b.setAttribute("aria-pressed", String(j === i)); });
    ex.load(e.video, "Flicker Fusion", { fps: e.fps, frames: e.frames, keepTime: false });
    var other = function (r, name) {
      return r ? " " + name + ": " + r.after + "% flicker, warp " + r.warp + ", structure " + r.structure.toFixed(2) + "." : "";
    };
    cap.textContent = e.title + ": " + e.note + ". Flicker Fusion: flicker " + e.before + "% → " + e.after + "%, warp error " +
      e.warp_before + " → " + e.warp_after + ", structure " + e.structure.toFixed(2) + "." +
      other(e.aio, "CVPR 2023 method") + other(e.ff, "CVPR 2026 method") + (e.caveat ? " " + e.caveat : "");
  }
  EXAMPLES.forEach(function (e, i) {
    var b = document.createElement("button");
    b.type = "button"; b.className = "ex-card"; b.setAttribute("aria-pressed", "false");
    var img = document.createElement("img"); img.src = e.poster; img.alt = ""; img.loading = "lazy";
    var t = document.createElement("b"); t.textContent = e.title;
    var n = document.createElement("span"); n.className = "num";
    var nb = document.createElement("span"); nb.className = "lamp"; nb.textContent = e.before + "%";
    var na = document.createElement("span"); na.className = "steady"; na.textContent = e.after + "%";
    n.appendChild(nb); n.appendChild(document.createTextNode(" → ")); n.appendChild(na);
    var k = document.createElement("span"); k.className = "k"; k.textContent = e.note + (e.caveat ? " · partial" : "");
    b.appendChild(img); b.appendChild(t); b.appendChild(n); b.appendChild(k);
    b.addEventListener("click", function () { pick(i); });
    cards.appendChild(b); buttons.push(b);
  });
  pick(0);
})();
