"""
Component 04 — targeted MIMIC-IV-ECG waveform fetch.

The full waveform release is ~144 GB.  We do not need it.  The binding
constraint in this component is STEMI-vs-NSTEMI separation, which is decided by
ST-segment elevation measured in millivolts — information that exists only in
the raw signal.  That question lives entirely inside the ACS patients, and only
for ECGs recorded within the disclosure window.

    tier 1  ACS patients, ECGs in [T0-1h, T0+H]        13,097 studies  ~2.4 GB
    tier 2  tier 1 + a matched No_ACS sample           52,388 studies  ~9.4 GB
    tier 3  every study in the cohort window          146,367 studies ~26.3 GB

Tier 1 is the point; tier 2 only matters if waveform features should also feed
the four-class model.

Usage
-----
    python ecg_fetch.py --manifest-only       # plan and size, download nothing
    python ecg_fetch.py --tier 1              # fetch ~2.4 GB
    python ecg_fetch.py --tier 1 --workers 8  # parallel (default)
    python ecg_fetch.py --tier 2 --max-gb 6   # stop at a size cap

Credentials are prompted for, never stored, and never passed as a command-line
argument — shell history is readable and persists.  A credentialed PhysioNet
account with the MIMIC-IV-ECG DUA signed is required.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, DATA_DIR, LABEL_MAP, enable_utf8_stdout, save_json
from utils import banner, kv, section

enable_utf8_stdout()

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"
WAVE_DIR = os.path.join(DATA_DIR, "ecg_waveforms")
MANIFEST = os.path.join(DATA_DIR, "ecg_manifest.parquet")
KB_PER_STUDY = 180          # 10 s x 500 Hz x 12 leads, 16-bit, plus header
CR = chr(13)


# --------------------------------------------------------------------------
def build_manifest(tier: int, horizon: int | None = None,
                   no_acs_ratio: int = 3, seed: int = 42) -> pd.DataFrame:
    """Which study_ids are needed, and where they live on PhysioNet."""
    horizon = CFG.primary_horizon if horizon is None else horizon
    lookback = float(CFG.get("temporal.ecg_lookback_h", 1.0))
    raw = CFG.raw_dir

    master = pd.read_parquet(os.path.join(raw, "master_data.parquet"))
    master["intime"] = pd.to_datetime(master["intime"])
    master["acs_label"] = pd.to_numeric(
        master["acs_label"], errors="coerce").fillna(0).astype(int)
    rec = pd.read_parquet(os.path.join(raw, "ecg_records.parquet"))
    rec["ecg_time"] = pd.to_datetime(rec["ecg_time"])

    feats = pd.read_parquet(
        os.path.join(DATA_DIR, "features_H%d.parquet" % horizon),
        columns=["stay_id", "in_cohort"])
    cohort = set(feats.loc[feats.in_cohort == 1, "stay_id"])

    j = master[["subject_id", "stay_id", "intime", "acs_label"]].merge(
        rec, on="subject_id", how="inner")
    j["h"] = (j["ecg_time"] - j["intime"]).dt.total_seconds() / 3600.0
    j = j[(j["h"] >= -lookback) & (j["h"] <= horizon) & j["stay_id"].isin(cohort)]

    acs = j[j.acs_label > 0]
    if tier == 1:
        keep = acs
    elif tier == 2:
        neg = j[j.acs_label == 0]
        n = min(len(neg), acs.study_id.nunique() * no_acs_ratio)
        keep = pd.concat([acs, neg.sample(n, random_state=seed)])
    else:
        keep = j

    man = (keep.sort_values("h")
               .drop_duplicates("study_id")[["subject_id", "study_id",
                                             "stay_id", "acs_label", "h"]]
               .reset_index(drop=True))
    # PhysioNet layout: files/p1000/p10000032/s40689238/40689238.{hea,dat}
    sid = man["subject_id"].astype(str)
    man["path"] = ("files/p" + sid.str[:4] + "/p" + sid +
                   "/s" + man["study_id"].astype(str) + "/" +
                   man["study_id"].astype(str))
    return man


def report(man: pd.DataFrame) -> None:
    section("Manifest")
    kv("studies to fetch", format(len(man), ","))
    kv("unique patients", format(man.subject_id.nunique(), ","))
    kv("estimated size", "%.2f GB" % (len(man) * KB_PER_STUDY / 1e6))
    print()
    for k, name in LABEL_MAP.items():
        n = int((man.acs_label == k).sum())
        if n:
            kv("  " + name, format(n, ",") + " studies")


# --------------------------------------------------------------------------
def fetch(man: pd.DataFrame, user: str, pwd: str, max_gb: float | None,
          workers: int = 16) -> dict:
    """
    Download .hea + .dat per study, skipping anything already on disk.

    This transfer is latency-bound, not bandwidth-bound.  At 8 study-level
    workers it ran at 161 studies/min = 0.48 MB/s, which is nothing on a fibre
    link: the connection sits idle waiting on round-trips.  Two structural
    fixes, both implemented here:

      1. FILE-level work queue, not study-level.  Previously each worker did
         GET .hea -> wait -> GET .dat -> wait, serialising two full RTTs per
         study.  Flattening to 2N independent file tasks removes that stall and
         roughly doubles achievable concurrency for the same worker count.
      2. Higher concurrency with connection reuse.  A per-thread Session with a
         matched pool keeps TLS handshakes amortised; on a latency-bound
         workload throughput scales close to linearly with in-flight requests
         until the server or the link becomes the limit.

    Politeness is not abandoned: transient 429/503 responses trigger
    exponential backoff and a global cooldown, so if PhysioNet pushes back we
    slow down rather than retrying harder.  PhysioNet is a shared academic
    resource; the aim is to stop idling, not to monopolise it.
    """
    import random

    import requests

    os.makedirs(WAVE_DIR, exist_ok=True)
    lock = threading.Lock()
    st = {"files_ok": 0, "files_skip": 0, "files_fail": 0, "bytes": 0,
          "done": 0, "auth_failed": False, "stop": False, "throttled": 0,
          "cooldown_until": 0.0}
    t0 = time.time()
    budget = None if max_gb is None else max_gb * 1e9
    local = threading.local()

    # ---- flatten to file-level tasks -----------------------------------
    tasks = []
    for row in man.itertuples(index=False):
        stem = str(row.study_id)
        dd = os.path.join(WAVE_DIR, stem)
        for ext in (".hea", ".dat"):
            tasks.append((BASE + "/" + row.path + ext,
                          os.path.join(dd, stem + ext), dd))
    total = len(tasks)

    def session():
        if not hasattr(local, "s"):
            s = requests.Session()
            # preemptive Basic auth: PhysioNet answers unauthenticated
            # requests with 403 rather than a 401 challenge, so waiting for
            # the challenge never authenticates.
            token = base64.b64encode(("%s:%s" % (user, pwd)).encode()).decode()
            s.headers.update({"Authorization": "Basic " + token,
                              "User-Agent": "Mozilla/5.0 curl-compatible",
                              "Accept-Encoding": "identity",
                              "Connection": "keep-alive"})
            s.mount("https://", requests.adapters.HTTPAdapter(
                pool_connections=workers, pool_maxsize=workers,
                max_retries=0))
            local.s = s
        return local.s

    def one(task) -> None:
        url, dest, dd = task
        if st["stop"] or st["auth_failed"]:
            return
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            with lock:
                st["files_skip"] += 1
                st["done"] += 1
            return

        sess = session()
        for attempt in range(4):
            # respect a global cooldown if the server asked us to slow down
            wait = st["cooldown_until"] - time.time()
            if wait > 0:
                time.sleep(min(wait, 5.0))
            try:
                r = sess.get(url, timeout=(10, 90))
                if r.status_code == 401:
                    with lock:
                        st["auth_failed"] = True
                    return
                if r.status_code in (429, 500, 502, 503, 504):
                    back = (2 ** attempt) + random.random()
                    with lock:
                        st["throttled"] += 1
                        st["cooldown_until"] = max(st["cooldown_until"],
                                                   time.time() + back)
                    continue
                if r.status_code != 200:
                    break
                os.makedirs(dd, exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(r.content)
                with lock:
                    st["files_ok"] += 1
                    st["bytes"] += len(r.content)
                    st["done"] += 1
                    if budget is not None and st["bytes"] >= budget:
                        st["stop"] = True
                return
            except Exception:
                time.sleep((2 ** attempt) * 0.3 + random.random() * 0.2)
        with lock:
            st["files_fail"] += 1
            st["done"] += 1
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass

    def draw() -> None:
        d = st["done"]
        el = max(time.time() - t0, 1e-6)
        fresh = st["files_ok"]
        rate_files = fresh / el
        mbps = st["bytes"] / el / 1e6
        remaining = total - d
        eta = remaining / (d / el) if d else 0.0
        frac = d / total if total else 1.0
        bar = "#" * int(frac * 20) + "." * (20 - int(frac * 20))
        msg = ("  [%s] %s/%s files %5.1f%%  ok=%s skip=%s fail=%s thr=%s  "
               "%.2f GB  %.1f MB/s  %.0f files/min  eta %.0f min   ") % (
            bar, format(d, ","), format(total, ","), frac * 100,
            format(fresh, ","), format(st["files_skip"], ","),
            format(st["files_fail"], ","), format(st["throttled"], ","),
            st["bytes"] / 1e9, mbps, rate_files * 60, eta / 60)
        sys.stdout.write(CR + msg)
        sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, t) for t in tasks]
        for n, _ in enumerate(as_completed(futures), 1):
            if n <= 40 or n % 25 == 0 or n == total:
                draw()
            if st["auth_failed"]:
                break
    draw()
    print()

    if st["auth_failed"]:
        print("  [AUTH FAILED] check your PhysioNet username and password, and")
        print("  that you have signed the MIMIC-IV-ECG data use agreement.")
        return {"aborted": "auth"}
    if st["stop"]:
        print("  [STOP] size cap of %s GB reached - rerun to continue; "
              "downloaded files are skipped." % max_gb)
    if st["throttled"]:
        print("  note: %d requests were throttled by the server and retried "
              "with backoff." % st["throttled"])
        print("  If that number is large, rerun with fewer --workers.")

    return {"files_downloaded": st["files_ok"], "files_skipped": st["files_skip"],
            "files_failed": st["files_fail"], "throttled": st["throttled"],
            "gigabytes": st["bytes"] / 1e9,
            "minutes": (time.time() - t0) / 60, "workers": workers}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Targeted MIMIC-IV-ECG fetch")
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                    help="1=ACS only (~2.4GB), 2=+No_ACS sample, 3=whole cohort")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--max-gb", type=float, default=None,
                    help="stop after roughly this many GB (resumable)")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel FILE downloads (default 16); latency-bound, "
                         "so 32-48 is reasonable on fibre. Server push-back is "
                         "handled by automatic backoff.")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--user", default=None, help="PhysioNet username")
    a = ap.parse_args()

    banner("MIMIC-IV-ECG TARGETED FETCH - tier %d" % a.tier)
    man = build_manifest(a.tier, a.horizon)
    report(man)
    man.to_parquet(MANIFEST, index=False)
    print()
    kv("manifest saved", MANIFEST)
    kv("destination", WAVE_DIR)

    need = len(man) * KB_PER_STUDY / 1e6
    if a.manifest_only:
        print("\n  --manifest-only: nothing downloaded.")
        kv("free space needed", "%.2f GB" % need)
        return

    free_gb = None
    try:
        import shutil
        free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
        kv("free disk space", "%.1f GB" % free_gb)
    except Exception:
        pass
    if free_gb is not None and free_gb < need * 1.2:
        print("\n  [WARNING] need about %.1f GB but only %.1f GB free."
              % (need, free_gb))
        print("  Use --max-gb to cap the download, or --tier 1 for the minimum.")
        if input("  Continue anyway? (y/n): ").strip().lower() != "y":
            return

    section("PhysioNet credentials")
    print("  Used for this session only; never written to disk.")
    user = a.user or input("  username: ").strip()
    pwd = getpass.getpass("  password: ")

    section("Downloading (%d parallel workers)" % a.workers)
    stats = fetch(man, user, pwd, a.max_gb, workers=a.workers)
    save_json(stats, os.path.join(DATA_DIR, "ecg_fetch_stats.json"))
    section("Done")
    for k, v in stats.items():
        kv(k, "%.2f" % v if isinstance(v, float) else v)
    print("\n  Next: python ecg_waveform.py   (extract ST-segment features)")


if __name__ == "__main__":
    main()
