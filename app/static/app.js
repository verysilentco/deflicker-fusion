(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var HELP = {
    fusion: "Today's method. Best on still or slow scenes (tripod, lamps, the shooting range).",
    fusion_adaptive: "New. Same result on still scenes; on moving people it stops borrowing from frames " +
      "that do not line up and rebuilds blown-out skin, so no patches or halos."
  };
  var LABEL = { fusion: "Fusion", fusion_adaptive: "Motion-adaptive" };
  var method = "fusion_adaptive", file = null, jobs = [], selected = null, shownKey = "", player = null, logTimer = null;
  try { method = localStorage.getItem("lab.method") || method; } catch (e) {}

  // ------------------------------------------------------------------ new run form
  var segs = Array.prototype.slice.call(document.querySelectorAll(".seg button"));
  function setMethod(m) {
    method = m;
    segs.forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.method === m)); });
    $("mhelp").textContent = HELP[m];
    try { localStorage.setItem("lab.method", m); } catch (e) {}
  }
  segs.forEach(function (b) { b.addEventListener("click", function () { setMethod(b.dataset.method); }); });
  setMethod(method);

  function canRun() { $("go").disabled = !(file || $("path").value.trim()); }
  function choose(f) {
    file = f || null;
    $("chosen").hidden = !file;
    $("chosen").textContent = file ? file.name + "  ·  " + (file.size / 1e6).toFixed(1) + " MB" : "";
    if (file) $("path").value = "";
    canRun();
  }
  var drop = $("drop");
  drop.addEventListener("click", function () { $("file").click(); });
  drop.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
  $("file").addEventListener("change", function () { choose($("file").files[0]); });
  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); });
  });
  drop.addEventListener("drop", function (e) { if (e.dataTransfer.files.length) choose(e.dataTransfer.files[0]); });
  $("path").addEventListener("input", function () { if ($("path").value.trim()) choose(null); canRun(); });

  function query() {
    var q = "method=" + method;
    if ($("qon").checked) q += "&max_frames=" + Math.max(30, parseInt($("qn").value, 10) || 240);
    return q;
  }
  $("go").addEventListener("click", function () {
    $("msg").textContent = "";
    $("go").disabled = true;
    if (file) {
      var xhr = new XMLHttpRequest(), bar = $("ubar");
      xhr.open("POST", "/api/upload?name=" + encodeURIComponent(file.name) + "&" + query());
      bar.hidden = false;
      xhr.upload.onprogress = function (e) { if (e.lengthComputable) bar.firstChild.style.width = (100 * e.loaded / e.total) + "%"; };
      xhr.onload = function () {
        bar.hidden = true; bar.firstChild.style.width = "0";
        var r = JSON.parse(xhr.responseText || "{}");
        if (xhr.status !== 200) { $("msg").textContent = r.error || "Upload failed"; canRun(); return; }
        choose(null); select(r.id); refresh();
      };
      xhr.onerror = function () { bar.hidden = true; $("msg").textContent = "Upload failed"; canRun(); };
      xhr.send(file);
    } else {
      fetch("/api/path?" + query(), { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: $("path").value.trim() }) })
        .then(function (r) { return r.json().then(function (j) { return [r.ok, j]; }); })
        .then(function (res) {
          if (!res[0]) { $("msg").textContent = res[1].error || "Could not start"; canRun(); return; }
          $("path").value = ""; canRun(); select(res[1].id); refresh();
        });
    }
  });

  // ------------------------------------------------------------------ jobs list
  function pct(v) { return (Math.round(v * 10) / 10).toFixed(1) + "%"; }
  function statusLine(j) {
    if (j.status === "done" && j.report) return null;
    if (j.status === "running") {
      var p = j.progress && j.progress[1] ? " · part " + Math.min(j.progress[0] + 1, j.progress[1]) + "/" + j.progress[1] : "";
      return j.stage + p;
    }
    return { queued: "waiting for the GPU", preparing: "preparing", failed: "failed", cancelled: "cancelled" }[j.status] || j.status;
  }
  function renderList() {
    var box = $("jobs");
    if (!jobs.length) { box.innerHTML = '<p class="empty">No runs yet.</p>'; return; }
    box.textContent = "";
    jobs.forEach(function (j) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "job"; b.setAttribute("aria-current", String(j.id === selected));
      var img = document.createElement("img"); img.src = "/files/" + j.id + "/thumb.jpg"; img.alt = "";
      var d = document.createElement("div");
      var t = document.createElement("b"); t.textContent = j.name;
      var s = document.createElement("small");
      var line = statusLine(j);
      if (line === null) {
        var a = document.createElement("span"); a.className = "num lamp"; a.textContent = pct(j.report.flicker_before_pct);
        var c = document.createElement("span"); c.className = "num steady"; c.textContent = pct(j.report.flicker_after_pct);
        s.appendChild(a); s.appendChild(document.createTextNode(" → ")); s.appendChild(c);
        s.appendChild(document.createTextNode(" · " + LABEL[j.method]));
      } else {
        s.textContent = LABEL[j.method] + " · " + line;
      }
      d.appendChild(t); d.appendChild(s);
      b.appendChild(img); b.appendChild(d);
      b.addEventListener("click", function () { select(j.id); });
      box.appendChild(b);
    });
  }
  function select(id) { selected = id; shownKey = ""; renderList(); renderMain(); }

  // ------------------------------------------------------------------ main view
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function fmtTime(s) { s = Math.round(s); return s < 90 ? s + " s" : Math.floor(s / 60) + " min " + (s % 60) + " s"; }
  function post(url) { return fetch(url, { method: "POST" }).then(function (r) { return r.json(); }); }

  function renderMain() {
    var j = jobs.filter(function (x) { return x.id === selected; })[0];
    var main = $("main");
    var key = j ? j.id + "|" + j.status + "|" + j.stage + "|" + (j.progress || []).join("/") : "none";
    if (key === shownKey) return;
    var keepPlayer = j && j.status === "done" && shownKey.indexOf(j.id + "|done") === 0;
    shownKey = key;
    if (keepPlayer) return;
    if (player) { player.destroy(); player = null; }
    clearInterval(logTimer);
    main.textContent = "";
    if (!j) { main.appendChild(el("div", "card empty", "Drop a clip on the left to start. Finished runs open here with a before / after player.")); return; }

    var head = el("div", "head");
    head.appendChild(el("h2", null, j.name));
    head.appendChild(el("span", "chip", j.method_label + (j.max_frames ? " · first " + j.max_frames + " frames" : "")));
    main.appendChild(head);

    if (j.status === "done" && j.report) {
      var r = j.report;
      var card = el("div", "card");
      var stage = el("div", "stage");
      stage.appendChild(el("canvas"));
      stage.appendChild(el("span", "tag l", "Original"));
      stage.appendChild(el("span", "tag r", "Deflickered"));
      stage.setAttribute("aria-label", "Before and after. Drag to move the divider; left and right arrow keys work too.");
      card.appendChild(stage);
      var ctl = el("div", "ctl");
      var bPlay = el("button", null, "Pause"), bPrev = el("button", null, "‹ frame"), bNext = el("button", null, "frame ›");
      ctl.appendChild(bPlay); ctl.appendChild(bPrev); ctl.appendChild(bNext);
      var rates = [0.25, 0.5, 1].map(function (v) {
        var b = el("button", null, v === 0.25 ? "¼×" : v === 0.5 ? "½×" : "1×"); b.dataset.rate = v;
        b.setAttribute("aria-pressed", String(v === 0.25)); ctl.appendChild(b); return b;
      });
      ctl.appendChild(el("span", "sp"));
      var bWipe = el("button", null, "Wipe"), bSbs = el("button", null, "Side by side");
      bWipe.setAttribute("aria-pressed", "true"); bSbs.setAttribute("aria-pressed", "false");
      ctl.appendChild(bWipe); ctl.appendChild(bSbs);
      var fr = el("span", "fr", "frame –"); ctl.appendChild(fr);
      card.appendChild(ctl);
      main.appendChild(card);
      player = makePlayer(stage, "/files/" + j.id + "/" + j.outputs.preview, r.fps, r.frames,
        { play: bPlay, prev: bPrev, next: bNext, rates: rates, wipe: bWipe, sbs: bSbs, frame: fr });

      var stats = el("div", "stats");
      function stat(k, v, s, cls) {
        var d = el("div", "stat"); d.appendChild(el("div", "k", k));
        var vv = el("div", "v"); if (typeof v === "string") vv.textContent = v; else vv.appendChild(v);
        if (cls) vv.className += " " + cls; d.appendChild(vv); if (s) d.appendChild(el("div", "s", s)); stats.appendChild(d);
      }
      var fl = el("span");
      fl.appendChild(el("span", "lamp", pct(r.flicker_before_pct))); fl.appendChild(document.createTextNode(" → "));
      fl.appendChild(el("span", "steady", pct(r.flicker_after_pct)));
      stat("Flicker", fl, "FlickerScope, before → after");
      var hz = r.chunks.map(function (c) { return c.hz; }), per = r.chunks.some(function (c) { return c.periodic; });
      var lo = Math.min.apply(null, hz), hi = Math.max.apply(null, hz);
      stat("Flicker rate", (hi - lo < 0.05 ? lo.toFixed(2) : lo.toFixed(1) + "–" + hi.toFixed(1)) + " Hz",
        (per ? "regular (lamp-like)" : "irregular") + ", on playback");
      var ev = {}; r.chunks.forEach(function (c) { (c.events || []).forEach(function (t) { ev[t] = 1; }); });
      stat("Clip", r.frames + " frames", r.fps.toFixed(2) + " fps · " + Object.keys(ev).length + " flash/event frames kept");
      stat("Time", fmtTime(r.seconds_this_run || 0), ((r.frames / Math.max(1, r.seconds_this_run || 1))).toFixed(1) + " frames per second");
      main.appendChild(stats);

      var act = el("div", "actions");
      [["Deflickered video", j.outputs.deflickered], ["Side-by-side video", j.outputs.compare], ["Report (JSON)", j.outputs.report]]
        .forEach(function (x) { var a = el("a", null, "⬇ " + x[0]); a.href = "/files/" + j.id + "/" + encodeURIComponent(x[1]); a.download = x[1]; act.appendChild(a); });
      var other = j.method === "fusion" ? "fusion_adaptive" : "fusion";
      var re = el("button", null, "Run again with " + LABEL[other]);
      re.addEventListener("click", function () {
        post("/api/jobs/" + j.id + "/rerun?method=" + other + (j.max_frames ? "&max_frames=" + j.max_frames : ""))
          .then(function (n) { select(n.id); refresh(); });
      });
      act.appendChild(re);
      if (j.max_frames) {
        var full = el("button", null, "Run the whole clip");
        full.addEventListener("click", function () {
          post("/api/jobs/" + j.id + "/rerun?method=" + j.method).then(function (n) { select(n.id); refresh(); });
        });
        act.appendChild(full);
      }
      var del = el("button", "danger", "Delete");
      del.addEventListener("click", function () {
        if (!confirm("Delete this run and its videos?")) return;
        post("/api/jobs/" + j.id + "/delete").then(function () { selected = null; refresh(); });
      });
      act.appendChild(del);
      main.appendChild(act);
      return;
    }

    var box = el("div", "card progress");
    if (j.status === "failed" || j.status === "cancelled") {
      box.appendChild(el("div", "big", j.status === "failed" ? "This run failed" : "Cancelled"));
      if (j.error) box.appendChild(el("pre", "log", j.error));
      var a2 = el("div", "actions"); a2.style.marginTop = "12px";
      var retry = el("button", null, "Run again");
      retry.addEventListener("click", function () {
        post("/api/jobs/" + j.id + "/rerun?method=" + j.method + (j.max_frames ? "&max_frames=" + j.max_frames : ""))
          .then(function (n) { select(n.id); refresh(); });
      });
      var del2 = el("button", "danger", "Delete");
      del2.addEventListener("click", function () { post("/api/jobs/" + j.id + "/delete").then(function () { selected = null; refresh(); }); });
      a2.appendChild(retry); a2.appendChild(del2); box.appendChild(a2);
      main.appendChild(box);
      return;
    }
    var p = j.progress || [0, 0];
    box.appendChild(el("div", "big", j.status === "running" ? j.stage : statusLine(j)));
    var sub = j.status === "running"
      ? (p[1] ? "part " + Math.min(p[0] + 1, p[1]) + " of " + p[1] + " · " : "") + "running " + fmtTime(Date.now() / 1000 - j.started)
      : "Starts when the GPU is free.";
    box.appendChild(el("div", "help", sub + (j.frames ? " · " + j.frames + " frames @ " + j.fps.toFixed(2) + " fps" : "")));
    var bar = el("div", "bar"); var fill = el("i"); bar.appendChild(fill); box.appendChild(bar);
    fill.style.width = (p[1] ? 100 * p[0] / p[1] : 2) + "%";
    var a3 = el("div", "actions"); a3.style.marginTop = "12px";
    var cancel = el("button", "danger", "Cancel");
    cancel.addEventListener("click", function () { post("/api/jobs/" + j.id + "/cancel").then(refresh); });
    a3.appendChild(cancel); box.appendChild(a3);
    var det = el("details"); det.appendChild(el("summary", null, "Log"));
    var pre = el("pre", "log", ""); det.appendChild(pre); box.appendChild(det);
    main.appendChild(box);
    function pull() { fetch("/api/log/" + j.id).then(function (r) { return r.json(); }).then(function (x) { pre.textContent = x.log; }); }
    pull(); logTimer = setInterval(pull, 2500);
  }

  function refresh() {
    return fetch("/api/jobs", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (list) {
      jobs = list;
      if (selected && !jobs.some(function (j) { return j.id === selected; })) selected = null;
      if (!selected && jobs.length) selected = jobs[0].id;
      renderList(); renderMain();
    }).catch(function () {});
  }
  refresh();
  setInterval(refresh, 1500);

  // ------------------------------------------------------------------ before / after player
  // One side-by-side video (original | deflickered) drawn on a canvas, so the halves can never drift.
  function makePlayer(stage, src, fps, frames, ui) {
    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    var cv = stage.querySelector("canvas"), ctx = cv.getContext("2d");
    var v = document.createElement("video");
    v.src = src; v.muted = true; v.loop = true; v.playsInline = true; v.preload = "auto"; v.setAttribute("playsinline", "");
    $("pool").appendChild(v);
    var wipe = 0.5, target = null, vel = 0, dragging = false, grab = 0, W = 1, H = 1, DPR = 1, sbs = false, alive = true, rate = 0.25;
    v.playbackRate = rate;
    v.addEventListener("loadedmetadata", function () {
      stage.style.aspectRatio = (sbs ? v.videoWidth : v.videoWidth / 2) + " / " + v.videoHeight; resize();
      v.play().catch(function () {});
    });
    function resize() { var r = stage.getBoundingClientRect(); DPR = window.devicePixelRatio || 1; W = r.width; H = r.height; cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR); }
    var ro = window.ResizeObserver ? new ResizeObserver(resize) : null; if (ro) ro.observe(stage);
    function syncPlay() { ui.play.textContent = v.paused ? "Play" : "Pause"; }
    v.addEventListener("play", syncPlay); v.addEventListener("pause", syncPlay);
    ui.play.addEventListener("click", function () { if (v.paused) v.play().catch(function () {}); else v.pause(); });
    function step(d) { v.pause(); var n = Math.max(0, Math.min(frames - 1, Math.floor(v.currentTime * fps) + d)); v.currentTime = (n + 0.5) / fps; }
    ui.prev.addEventListener("click", function () { step(-1); }); ui.next.addEventListener("click", function () { step(1); });
    ui.rates.forEach(function (b) { b.addEventListener("click", function () {
      rate = +b.dataset.rate; v.playbackRate = rate; ui.rates.forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
    }); });
    function setMode(s) {
      sbs = s; stage.classList.toggle("sbs", s);
      ui.wipe.setAttribute("aria-pressed", String(!s)); ui.sbs.setAttribute("aria-pressed", String(s));
      if (v.videoWidth) stage.style.aspectRatio = (s ? v.videoWidth : v.videoWidth / 2) + " / " + v.videoHeight;
      resize();
    }
    ui.wipe.addEventListener("click", function () { setMode(false); }); ui.sbs.addEventListener("click", function () { setMode(true); });
    function fit(sw, sh) { var s = Math.min(W / sw, H / sh); return { x: (W - sw * s) / 2, y: (H - sh * s) / 2, w: sw * s, h: sh * s }; }
    var last = performance.now();
    function tick(now) {
      if (!alive) return;
      var dt = Math.min(1 / 30, (now - last) / 1000); last = now;
      if (target !== null) {   // critically damped spring, interruptible
        var k = 246.74, c = 31.42, n = 4, h = dt / n;
        for (var i = 0; i < n; i++) { var a = -k * (wipe - target) - c * vel; vel += a * h; wipe += vel * h; }
        if (Math.abs(wipe - target) < 0.001 && Math.abs(vel) < 0.01) { wipe = target; vel = 0; target = null; }
      }
      draw(); requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    function draw() {
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
      if (v.readyState < 2) return;
      var vw = v.videoWidth, vh = v.videoHeight, half = vw / 2;
      if (sbs) { var g = fit(vw, vh); ctx.drawImage(v, 0, 0, vw, vh, g.x, g.y, g.w, g.h); }
      else {
        var f = fit(half, vh), x = f.x + wipe * f.w;
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, H); ctx.clip(); ctx.drawImage(v, 0, 0, half, vh, f.x, f.y, f.w, f.h); ctx.restore();
        ctx.save(); ctx.beginPath(); ctx.rect(x, 0, W - x, H); ctx.clip(); ctx.drawImage(v, half, 0, half, vh, f.x, f.y, f.w, f.h); ctx.restore();
        ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fillRect(x - 2, 0, 4, H); ctx.fillStyle = "#fff"; ctx.fillRect(x - 1, 0, 2, H);
        var ky = H / 2; ctx.beginPath(); ctx.arc(x, ky, 16, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 2; ctx.beginPath();
        ctx.moveTo(x - 5, ky - 5); ctx.lineTo(x - 9, ky); ctx.lineTo(x - 5, ky + 5);
        ctx.moveTo(x + 5, ky - 5); ctx.lineTo(x + 9, ky); ctx.lineTo(x + 5, ky + 5); ctx.stroke();
      }
      ui.frame.textContent = "frame " + (Math.min(frames - 1, Math.floor(v.currentTime * fps)) + 1) + " / " + frames;
    }
    function px(e) { return e.clientX - stage.getBoundingClientRect().left; }
    function geom() { var f = fit(v.videoWidth / 2 || 16, v.videoHeight || 9); return f; }
    stage.addEventListener("pointerdown", function (e) {
      if (sbs) return;
      var f = geom(); stage.setPointerCapture(e.pointerId); dragging = true; target = null; vel = 0;
      var lineX = f.x + wipe * f.w, x = px(e);
      grab = Math.abs(x - lineX) < 28 ? x - lineX : 0;
      var to = Math.max(0, Math.min(1, (x - f.x) / f.w));
      if (!grab) { if (reduce) wipe = to; else target = to; }
    });
    stage.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      var f = geom(), want = Math.max(0, Math.min(1, (px(e) - grab - f.x) / f.w));
      if (target === null) wipe = want; else if (Math.abs(target - wipe) < 0.02) { target = null; wipe = want; } else target = want;
    });
    function end() { dragging = false; }
    stage.addEventListener("pointerup", end); stage.addEventListener("pointercancel", end);
    stage.tabIndex = 0;
    stage.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { wipe = Math.max(0, wipe - .05); e.preventDefault(); }
      if (e.key === "ArrowRight") { wipe = Math.min(1, wipe + .05); e.preventDefault(); }
      if (e.key === " ") { e.preventDefault(); ui.play.click(); }
      if (e.key === ",") step(-1);
      if (e.key === ".") step(1);
    });
    return { destroy: function () { alive = false; if (ro) ro.disconnect(); v.pause(); v.removeAttribute("src"); v.load(); v.remove(); } };
  }
})();
