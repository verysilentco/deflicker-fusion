"""Download the benchmark's frames from the Blind Video Deflickering dataset (Lei et al., CVPR 2023).

The dataset (CC BY 4.0) is published on Hugging Face as one large zip:
https://huggingface.co/datasets/xrenaa/Blind-Video-Deflickering-Dataset
Only the folders the benchmark uses are fetched (the slow-motion set and four time-lapse clips), with
HTTP range requests, so the whole archive is never downloaded. Files already on disk are skipped, so an
interrupted run simply continues.

python fetch_dataset.py                 -> DATA/All-In-One-Deflicker-Dataset/...   (DATA: DEFLICKER_DATA, default ./data)
python fetch_dataset.py --out somewhere -> somewhere/All-In-One-Deflicker-Dataset/...
Then: python prep.py (point it at a different location with DEFLICKER_DATASET or clips.json).
"""
import argparse
import os
import time

from common import DATA

URL = ("https://huggingface.co/datasets/xrenaa/Blind-Video-Deflickering-Dataset/resolve/main/"
       "All-In-One-Deflicker-Dataset.zip")
WANT = ["All-In-One-Deflicker-Dataset/SlowMotion/"] + \
       [f"All-In-One-Deflicker-Dataset/TimeLapse/Youtube/{k}/" for k in ["sunset", "city", "snownight", "night2day"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA, help="folder that receives All-In-One-Deflicker-Dataset/")
    ap.add_argument("--attempts", type=int, default=8)
    a = ap.parse_args()
    from remotezip import RemoteZip  # pip install remotezip
    out = os.path.abspath(a.out)
    for attempt in range(a.attempts):
        try:
            with RemoteZip(URL, timeout=60) as z:
                sel = [i for i in z.infolist() if not i.is_dir() and any(i.filename.startswith(w) for w in WANT)
                       and i.filename.lower().endswith((".jpg", ".png"))]
                todo = [i for i in sel if not (os.path.exists(os.path.join(out, i.filename))
                                               and os.path.getsize(os.path.join(out, i.filename)) == i.file_size)]
                print(f"attempt {attempt}: {len(todo)}/{len(sel)} files remaining", flush=True)
                if not todo:
                    break
                for i in todo:
                    z.extract(i, out)
        except Exception as e:  # flaky connections: reopen and continue where it stopped
            print("retry after error:", type(e).__name__, flush=True)
            time.sleep(3)
    print("done:", os.path.join(out, "All-In-One-Deflicker-Dataset"))


if __name__ == "__main__":
    main()
