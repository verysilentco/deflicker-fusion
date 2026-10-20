"""Deflicker Lab: a local page for running your own clips through FlickerScope + Flicker Fusion.

python app/server.py            -> http://127.0.0.1:8770/   (PORT=... to change the port)

Drop a video (or paste a path to one on this computer), pick the method, and it runs on this machine's
GPU through bench/deflicker_file.py, one job at a time. Each job keeps its own folder under
app/jobs/<id>/: the deflickered video (with the original audio), a side-by-side compare video,
a web preview for the before/after player, and the flicker report.

Standard library only. Binds to 127.0.0.1, so it is reachable from this computer alone.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

APP = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(APP), "bench")
STATIC = os.path.join(APP, "static")
JOBS = os.path.join(APP, "jobs")
PORT = int(os.environ.get("PORT", "8770"))
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
METHODS = {"fusion": "Flicker Fusion", "fusion_adaptive": "Flicker Fusion, motion-adaptive (new)"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png",
         ".mov": "video/quicktime", ".svg": "image/svg+xml"}

os.makedirs(JOBS, exist_ok=True)
lock = threading.Lock()
wake = threading.Event()
running = {}  # job id -> Popen


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


FF = ffmpeg()


def job_path(jid, *parts):
    if not re.fullmatch(r"[0-9a-f]{12}", jid or ""):
        raise ValueError("bad job id")
    return os.path.join(JOBS, jid, *parts)


def load(jid):
    with open(job_path(jid, "job.json"), encoding="utf-8") as f:
        return json.load(f)


def save(job):
    p = job_path(job["id"], "job.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(job, f, indent=1)
    os.replace(p + ".tmp", p)


def all_jobs():
    out = []
    for jid in os.listdir(JOBS):
        try:
            out.append(load(jid))
        except Exception:
            continue
    return sorted(out, key=lambda j: j["created"], reverse=True)


def thumbnail(src, dst):
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-frames:v", "1", "-vf", "scale=360:-2", dst])


def new_job(src, name, method, max_frames, copied, hold=False):
    """hold: the input is not in its final place yet; call enqueue() once it is."""
    jid = uuid.uuid4().hex[:12]
    os.makedirs(job_path(jid))
    job = dict(id=jid, name=name, input=src, copied=copied, method=method, method_label=METHODS[method],
               max_frames=max_frames, status="preparing" if hold else "queued", created=time.time(), started=None,
               finished=None, stage="waiting for the GPU", progress=[0, 0], report=None, error=None, outputs={})
    save(job)
    thumbnail(src, job_path(jid, "thumb.jpg"))
    if not hold:
        wake.set()
    return job


def enqueue(job, **changes):
    job.update(changes, status="queued")
    save(job)
    wake.set()
    return job


# ----------------------------------------------------------------------------------------- worker
def run_job(job):
    jid = job["id"]
    out_dir = job_path(jid)
    stem = os.path.splitext(os.path.basename(job["input"]))[0]
    cmd = [sys.executable, "-u", os.path.join(BENCH, "deflicker_file.py"), job["input"], "--out", out_dir]
    if job["method"] == "fusion_adaptive":
        cmd.append("--adaptive")
    if job.get("max_frames"):
        cmd += ["--max-frames", str(int(job["max_frames"]))]
    job.update(status="running", started=time.time(), stage="starting")
    save(job)
    log = open(job_path(jid, "log.txt"), "w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")  # library deprecation chatter
    p = subprocess.Popen(cmd, cwd=BENCH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    running[jid] = p
    for line in p.stdout:
        log.write(line)
        log.flush()
        line = line.rstrip()
        with lock:
            job = load(jid)
            if line.startswith("@stage "):
                job["stage"] = line[7:]
            elif line.startswith("@progress "):
                a, b = line[10:].split("/")
                job["progress"] = [int(a), int(b)]
            elif m := re.match(r"\S.*: (\d+) frames @ ([\d.]+) fps", line):
                job["frames"], job["fps"] = int(m.group(1)), float(m.group(2))
            save(job)
    code = p.wait()
    running.pop(jid, None)
    log.close()
    job = load(jid)
    if job["status"] == "cancelled":
        return
    rep_path = os.path.join(out_dir, f"{stem}_report.json")
    if code != 0 or not os.path.exists(rep_path):
        tail = open(job_path(jid, "log.txt"), encoding="utf-8", errors="replace").read().splitlines()[-12:]
        job.update(status="failed", finished=time.time(), stage="failed", error="\n".join(tail))
        save(job)
        return
    job["stage"] = "making the web preview"
    save(job)
    cmp_path = os.path.join(out_dir, f"{stem}_compare.mp4")
    # browsers decode up to ~4K wide; the full-resolution compare video of a 4K clip is 7680 wide
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", cmp_path, "-vf", "scale='min(1920,iw)':-2", "-an",
                    "-c:v", "libx264", "-crf", "21", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", job_path(jid, "preview.mp4")])
    report = json.load(open(rep_path, encoding="utf-8"))
    job.update(status="done", finished=time.time(), stage="done", report=report,
               outputs=dict(deflickered=f"{stem}_deflickered.mp4", compare=f"{stem}_compare.mp4",
                            report=f"{stem}_report.json", preview="preview.mp4"))
    save(job)


def worker():
    # jobs left "running" by a previous server process go back to the queue (the tool resumes them)
    for job in all_jobs():
        if job["status"] == "running":
            job.update(status="queued", stage="waiting for the GPU (resuming)")
            save(job)
    while True:
        queued = [j for j in all_jobs() if j["status"] == "queued"]
        if not queued:
            wake.wait(2.0)
            wake.clear()
            continue
        job = min(queued, key=lambda j: j["created"])
        try:
            run_job(job)
        except Exception as e:  # never let one bad job stop the queue
            job = load(job["id"])
            job.update(status="failed", finished=time.time(), stage="failed", error=repr(e))
            save(job)


# ----------------------------------------------------------------------------------------- http
LOCAL_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def from_this_page(self, write: bool) -> bool:
        """Only the Lab's own page may use it: other websites open in the browser must not start, delete or
        read jobs (Host check stops DNS rebinding; Origin check stops cross-site POSTs)."""
        if self.headers.get("Host", "") not in LOCAL_HOSTS:
            return False
        origin = self.headers.get("Origin")
        return not (write or origin) or origin in {f"http://{h}" for h in LOCAL_HOSTS}

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        """Static file with HTTP Range support (video seeking needs it)."""
        if not os.path.isfile(path):
            return self.send_json({"error": "not found"}, 404)
        size = os.path.getsize(path)
        ctype = TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if not self.from_this_page(write=False):
            return self.send_json({"error": "forbidden"}, 403)
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        if not parts:
            return self.send_file(os.path.join(STATIC, "index.html"))
        if parts[0] == "static" and len(parts) == 2 and "/" not in parts[1] and ".." not in parts[1]:
            return self.send_file(os.path.join(STATIC, parts[1]))
        if parts == ["api", "jobs"]:
            return self.send_json(all_jobs())
        if len(parts) == 3 and parts[:2] == ["api", "log"]:
            try:
                txt = open(job_path(parts[2], "log.txt"), encoding="utf-8", errors="replace").read()
            except (OSError, ValueError):
                txt = ""
            return self.send_json({"log": "\n".join(l for l in txt.splitlines() if not l.startswith("@"))[-6000:]})
        if len(parts) == 3 and parts[0] == "files":
            name = parts[2]
            if "/" in name or "\\" in name or ".." in name:
                return self.send_json({"error": "bad name"}, 400)
            try:
                return self.send_file(job_path(parts[1], name))
            except ValueError:
                return self.send_json({"error": "bad id"}, 400)
        return self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.from_this_page(write=True):
            return self.send_json({"error": "forbidden"}, 403)
        url = urlparse(self.path)
        q = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        method = (q.get("method") or ["fusion"])[0]
        if method not in METHODS:
            return self.send_json({"error": "unknown method"}, 400)
        max_frames = int((q.get("max_frames") or ["0"])[0] or 0) or None
        if parts == ["api", "upload"]:
            # raw body upload: the file's bytes, its name in the query string
            name = os.path.basename((q.get("name") or ["clip.mp4"])[0])
            if not name.lower().endswith(VIDEO_EXT):
                return self.send_json({"error": "not a video file"}, 400)
            n = int(self.headers.get("Content-Length") or 0)
            tmp = os.path.join(JOBS, f"_upload_{uuid.uuid4().hex[:8]}_{name}")
            with open(tmp, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            job = new_job(tmp, name, method, max_frames, copied=True, hold=True)
            dst = job_path(job["id"], name)
            os.replace(tmp, dst)
            return self.send_json(enqueue(job, input=dst))
        if parts == ["api", "path"]:
            src = self.body_json().get("path", "").strip().strip('"')
            if not os.path.isfile(src) or not src.lower().endswith(VIDEO_EXT):
                return self.send_json({"error": "no video file at that path"}, 400)
            return self.send_json(new_job(os.path.abspath(src), os.path.basename(src), method, max_frames, copied=False))
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in ("cancel", "delete", "rerun"):
            jid, act = parts[2], parts[3]
            try:
                job = load(jid)
            except (OSError, ValueError):
                return self.send_json({"error": "no such job"}, 404)
            if act == "cancel":
                job.update(status="cancelled", stage="cancelled", finished=time.time())
                save(job)
                p = running.get(jid)
                if p:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True) if os.name == "nt" \
                        else p.kill()
                return self.send_json(job)
            if act == "rerun":
                new = new_job(job["input"], job["name"], method, max_frames, copied=job["copied"], hold=True)
                dst = job["input"]
                if job["copied"]:  # an uploaded clip lives in the old job's folder: give the new job its own
                    dst = job_path(new["id"], job["name"])
                    try:
                        os.link(job["input"], dst)
                    except OSError:
                        shutil.copy2(job["input"], dst)
                return self.send_json(enqueue(new, input=dst))
            if job["status"] == "running":
                return self.send_json({"error": "cancel it first"}, 409)
            shutil.rmtree(job_path(jid), ignore_errors=True)
            return self.send_json({"deleted": jid})
        return self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Deflicker Lab on http://127.0.0.1:{PORT}/", flush=True)
    srv.serve_forever()
