"""
Component 04 — expanded cardiac biomarker extraction from MIMIC-IV (PhysioNet).

Why
---
The original extraction returned Troponin T only: 52,487 results covering 20,441
of 203,016 ED stays.  Unstable angina is *defined* by a normal troponin, so a UA
patient with no troponin on record is indistinguishable from anyone else — which
is exactly what the data shows, with 89.47% of UA cases having a nearest
neighbour of a different class.  If BIDMC also assayed Troponin I, CK-MB or
myoglobin under different item labels, those results are sitting unused.

This module pulls them from the PhysioNet file server, since not everyone has
BigQuery access.

The size problem
----------------
`labevents.csv.gz` is ~2.4 GB compressed and ~17 GB uncompressed (158M rows).
We never write the uncompressed form: the gzip stream is decoded in chunks and
each chunk is filtered down to (a) cardiac biomarker itemids and (b) subject_ids
that appear in our cohort.  What lands on disk is a few tens of MB.

Usage
-----
    python labs_fetch.py --discover        # download the tiny dictionary only
    python labs_fetch.py                   # full run (downloads ~2.4 GB once)
    python labs_fetch.py --keep-archive    # keep labevents.csv.gz afterwards
"""
from __future__ import annotations

import argparse
import base64
import getpass
import gzip
import io
import os
import sys
import time

import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, DATA_DIR, enable_utf8_stdout, save_json
from utils import banner, df_to_markdown, kv, section

enable_utf8_stdout()

BASE = "https://physionet.org/files/mimiciv/3.1/hosp"
RAW_DIR = os.path.join(DATA_DIR, "mimic_raw")
OUT = os.path.join(DATA_DIR, "lab_values_expanded.parquet")
CR = chr(13)

# Cardiac markers worth having.  Troponin is the diagnostic standard; CK-MB
# rises faster and steeper in large infarcts, which is exactly the NSTEMI/STEMI
# question; myoglobin is the earliest riser and helps at short horizons.
PATTERNS = ("tropon", "ck-mb", "ck mb", "ckmb", "creatine kinase", "cpk",
            "myoglobin", "bnp", "natriuretic")


# --------------------------------------------------------------------------
def _curl_available() -> bool:
    import shutil
    return shutil.which("curl") is not None or shutil.which("curl.exe") is not None


def _download_cookie(url: str, name: str, cookie: str, dest: str,
                     have: int) -> str:
    """
    Fetch using the browser's PhysioNet session cookie.

    PhysioNet's web session and its HTTP Basic auth are separate paths, and on
    some accounts the file server rejects Basic while the logged-in session is
    perfectly authorised — which is why the browser can download a file the
    script cannot.  Reusing the session cookie sidesteps that entirely and,
    unlike a browser download, gives us resume and a progress bar.
    """
    import shutil
    import subprocess

    # Accept either "sessionid=abc123" or a bare "abc123".  Pasting just the
    # value from the browser's cookie inspector is the obvious mistake, and it
    # produces a malformed Cookie header that the server silently treats as no
    # session at all - indistinguishable from a permissions failure.
    cookie = cookie.strip()
    if "=" not in cookie:
        cookie = "sessionid=" + cookie
        kv("note", "cookie name was missing; using sessionid=<value>")

    exe = shutil.which("curl.exe") or shutil.which("curl")
    cmd = [exe, "--fail", "--location", "--progress-bar",
           "-H", "Cookie: " + cookie,
           "-H", "User-Agent: Mozilla/5.0",
           "-o", dest]
    if have:
        cmd += ["-C", "-"]
    cmd.append(url)

    kv("fetching", name + ("  (resuming at %.2f GB)" % (have / 1e9) if have else ""))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise PermissionError(
            "curl exited %d for %s\n"
            "  The session cookie may have expired - copy a fresh one from the\n"
            "  browser (F12 > Application > Cookies > physionet.org)."
            % (proc.returncode, url))

    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size < 1024:
        raise RuntimeError("downloaded %d bytes - error page, not data." % size)
    with open(dest, "rb") as fh:
        if fh.read(2) != b"\x1f\x8b":
            raise RuntimeError(
                "%s is not gzip - the cookie was probably rejected and this is\n"
                "  a saved login page. Delete it and copy a fresh cookie." % dest)
    kv("  received", "%.2f MB" % (size / 1e6))
    return dest


def _download_curl(url: str, name: str, user: str, pwd: str, dest: str,
                   have: int) -> str:
    """
    Fetch via curl, passing the password on stdin so it never appears in the
    process list or shell history (`-K -` reads config from stdin).
    """
    import shutil
    import subprocess

    exe = shutil.which("curl.exe") or shutil.which("curl")
    cfg = 'user = "%s:%s"\n' % (user, pwd)
    cmd = [exe, "-K", "-", "--fail", "--location", "--silent", "--show-error",
           "--progress-bar", "-o", dest]
    if have:
        cmd += ["-C", "-"]          # resume
    cmd.append(url)

    kv("fetching", name + ("  (resuming at %.2f GB)" % (have / 1e9) if have else ""))
    proc = subprocess.run(cmd, input=cfg, text=True, capture_output=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if "403" in err:
            raise PermissionError(
                "403 Forbidden for %s\n"
                "  Sign the data use agreement for MIMIC-IV 3.1 specifically at\n"
                "  https://physionet.org/content/mimiciv/3.1/ — the MIMIC-IV-ECG\n"
                "  agreement does not cover it." % url)
        if "401" in err:
            raise PermissionError("401 Unauthorized - credentials rejected.")
        raise RuntimeError("curl failed (exit %d): %s" % (proc.returncode, err))

    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size < 1024:
        raise RuntimeError(
            "downloaded %d bytes - that is an error page, not data." % size)
    kv("  received", "%.2f MB" % (size / 1e6))
    return dest


def download(name: str, user: str, pwd: str, dest: str,
             cookie: str | None = None) -> str:
    """
    Resumable download with a progress line.

    Two things matter for PhysioNet specifically:

      * No HEAD probe.  The file server answers HEAD inconsistently for these
        paths, and a failed probe was masking a perfectly good GET.  The size
        is read from the GET response headers instead.
      * A browser-like User-Agent.  The default `python-requests/x.y` string is
        rejected by some PhysioNet front-ends with 403 even when the
        credentials and DUA are fine — which is why `wget` works and a naive
        script does not.
    """
    import requests

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = BASE + "/" + name
    have = os.path.getsize(dest) if os.path.exists(dest) else 0

    # Already present and genuinely gzip?  Use it and skip the network.  This
    # is what makes a manual browser download work: drop the file in and run.
    # The magic-byte check matters because PhysioNet serves an HTML error page
    # with HTTP 200-looking success to curl without --fail, and a 12 KB
    # "<!DOCTYPE html>" masquerading as a .csv.gz is exactly the sort of thing
    # that wastes an evening.
    if have > 1024:
        with open(dest, "rb") as fh:
            magic = fh.read(2)
        if magic == b"\x1f\x8b":
            kv(name, "already present (%.2f MB) - skipping download"
                     % (have / 1e6))
            return dest
        raise RuntimeError(
            "%s exists but is not gzip (first bytes %r).\n"
            "  That is almost certainly a saved HTML error page. Delete it and\n"
            "  download again: %s" % (dest, magic, url))

    # curl first.  Empirically PhysioNet accepts curl and rejects requests with
    # 403 on these paths even with identical preemptive Basic auth headers, so
    # something in their front-end fingerprints the client beyond what we can
    # reproduce.  curl.exe ships with Windows 10+ and every mainstream Linux,
    # handles resume with -C -, and is what PhysioNet's own instructions tell
    # users to run.  We keep the requests path as a fallback rather than
    # pretending to know which header mattered.
    if cookie and _curl_available():
        return _download_cookie(url, name, cookie, dest, have)
    if _curl_available():
        return _download_curl(url, name, user, pwd, dest, have)

    # PREEMPTIVE Basic auth.  PhysioNet answers an unauthenticated request to
    # these paths with 403, not 401.  requests' `auth=` helper only attaches
    # credentials *after* a 401 challenge, so it never retries and the caller
    # sees a bare 403 that looks like a DUA problem.  curl -u works because it
    # sends the header on the first request; we do the same.
    token = base64.b64encode(("%s:%s" % (user, pwd)).encode()).decode()
    headers = {
        "Authorization": "Basic " + token,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) curl-compatible",
        "Accept": "*/*",
        "Accept-Encoding": "identity",
    }
    if have:
        headers["Range"] = "bytes=%d-" % have

    r = requests.get(url, headers=headers, stream=True,
                     timeout=(15, 180), allow_redirects=True)

    if r.status_code == 401:
        raise PermissionError(
            "401 Unauthorized - username or password rejected.")
    if r.status_code == 403:
        raise PermissionError(
            "403 Forbidden for %s\n"
            "  Credentials were accepted but this dataset is not authorised.\n"
            "  Open https://physionet.org/content/mimiciv/3.1/ while logged in\n"
            "  and confirm the data use agreement is signed for MIMIC-IV 3.1\n"
            "  specifically - the MIMIC-IV-ECG agreement does not cover it." % url)
    if r.status_code == 416 and have:
        kv(name, "already complete (%.2f GB)" % (have / 1e9))
        return dest
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0)) + (have if have else 0)
    if have and total and have >= total:
        kv(name, "already downloaded (%.2f GB)" % (have / 1e9))
        return dest

    mode = "ab" if r.status_code == 206 and have else "wb"
    if mode == "wb":
        have = 0
    t0, got = time.time(), have
    with open(dest, mode) as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            got += len(chunk)
            el = time.time() - t0
            speed = (got - have) / el / 1e6 if el else 0
            if total:
                frac = got / total
                bar = "#" * int(frac * 20) + "." * (20 - int(frac * 20))
                eta = (total - got) / ((got - have) / el) / 60 if el and got > have else 0
                msg = "  [%s] %5.1f%%  %.2f/%.2f GB  %.1f MB/s  eta %.0f min   " % (
                    bar, frac * 100, got / 1e9, total / 1e9, speed, eta)
            else:
                msg = "  %.2f GB  %.1f MB/s   " % (got / 1e9, speed)
            sys.stdout.write(CR + msg)
            sys.stdout.flush()
    print()
    return dest


# --------------------------------------------------------------------------
def discover(user: str, pwd: str, cookie: str | None = None) -> pd.DataFrame:
    """Which cardiac assays exist, and under what labels."""
    section("Lab dictionary (d_labitems.csv.gz — a few KB)")
    p = download("d_labitems.csv.gz", user, pwd,
                 os.path.join(RAW_DIR, "d_labitems.csv.gz"), cookie)
    d = pd.read_csv(p, compression="gzip")
    d.columns = [c.lower() for c in d.columns]
    lab = d["label"].fillna("").str.lower()
    mask = False
    for pat in PATTERNS:
        mask = mask | lab.str.contains(pat, regex=False)
    hits = d[mask].copy()
    section("Cardiac assays found in MIMIC-IV")
    print(df_to_markdown(hits[["itemid", "label", "fluid", "category"]]))
    return hits


def stream_filter(path: str, itemids: set, subjects: set | None,
                  chunk_rows: int = 2_000_000) -> pd.DataFrame:
    """
    Decode labevents.csv.gz in chunks, keep only rows we need.

    The uncompressed table is ~17 GB; it is never materialised.  Each chunk is
    filtered immediately and only the survivors are retained.
    """
    section("Streaming labevents.csv.gz (never uncompressed to disk)")
    keep = []
    total_rows = kept_rows = 0
    t0 = time.time()
    usecols = ["subject_id", "hadm_id", "itemid", "charttime",
               "value", "valuenum", "valueuom"]

    reader = pd.read_csv(path, compression="gzip", chunksize=chunk_rows,
                         usecols=lambda c: c.lower() in usecols,
                         low_memory=False)
    for i, chunk in enumerate(reader, 1):
        chunk.columns = [c.lower() for c in chunk.columns]
        total_rows += len(chunk)
        sel = chunk[chunk["itemid"].isin(itemids)]
        if subjects is not None and len(sel):
            sel = sel[sel["subject_id"].isin(subjects)]
        if len(sel):
            keep.append(sel)
            kept_rows += len(sel)
        el = time.time() - t0
        sys.stdout.write(CR + "  chunk %3d   scanned %s rows   kept %s   %.0f s   "
                         % (i, format(total_rows, ","), format(kept_rows, ","), el))
        sys.stdout.flush()
    print()
    if not keep:
        return pd.DataFrame(columns=usecols)
    return pd.concat(keep, ignore_index=True)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Expanded cardiac biomarker fetch")
    ap.add_argument("--discover", action="store_true",
                    help="only download the dictionary and list assays")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep labevents.csv.gz after filtering")
    ap.add_argument("--user", default=None)
    ap.add_argument("--cookie", default=None,
                    help="PhysioNet session cookie, e.g. \"sessionid=abc123\". "
                         "Use when Basic auth is rejected but the browser works.")
    a = ap.parse_args()

    banner("MIMIC-IV CARDIAC BIOMARKERS — PhysioNet file server")
    kv("source", BASE)
    kv("output", OUT)

    cookie = a.cookie
    if cookie:
        section("PhysioNet session cookie")
        print("  Reusing your browser session; no password needed.")
        user = pwd = ""
    else:
        section("PhysioNet credentials")
        print("  Used for this session only; never written to disk.")
        user = a.user or input("  username: ").strip()
        pwd = getpass.getpass("  password: ")

    hits = discover(user, pwd, cookie)
    if hits.empty:
        print("\n  No cardiac assays matched — check the patterns list.")
        return
    itemids = set(hits["itemid"].astype(int))
    kv("\n  itemids to extract", len(itemids))

    if a.discover:
        print("\n  --discover: stopping before the 2.4 GB download.")
        print("  Review the table above; if Troponin I / CK-MB appear with")
        print("  meaningful volume, rerun without --discover.")
        return

    # restrict to our cohort's patients to keep the output small
    subjects = None
    master_p = os.path.join(CFG.raw_dir, "master_data.parquet")
    if os.path.exists(master_p):
        subjects = set(pd.read_parquet(master_p, columns=["subject_id"])
                       ["subject_id"].dropna().astype(int))
        kv("cohort patients", format(len(subjects), ","))

    section("labevents.csv.gz  (~2.4 GB compressed, resumable)")
    path = download("labevents.csv.gz", user, pwd,
                    os.path.join(RAW_DIR, "labevents.csv.gz"), cookie)

    df = stream_filter(path, itemids, subjects)
    if df.empty:
        print("\n  Nothing matched — unexpected; check itemids.")
        return

    df = df.merge(hits[["itemid", "label", "fluid"]], on="itemid", how="left")
    df.rename(columns={"label": "lab_name"}, inplace=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df.to_parquet(OUT, index=False)

    section("Result")
    kv("rows kept", format(len(df), ","))
    kv("unique patients", format(df.subject_id.nunique(), ","))
    kv("saved", OUT)
    print()
    counts = (df.groupby("lab_name")
                .agg(rows=("valuenum", "size"),
                     patients=("subject_id", "nunique"),
                     median=("valuenum", "median"))
                .sort_values("rows", ascending=False).reset_index())
    print(df_to_markdown(counts))
    save_json({"rows": int(len(df)), "patients": int(df.subject_id.nunique()),
               "assays": counts.to_dict("records")},
              os.path.join(DATA_DIR, "labs_expanded_summary.json"))

    if not a.keep_archive and os.path.exists(path):
        size = os.path.getsize(path) / 1e9
        os.remove(path)
        kv("\n  removed archive", "labevents.csv.gz (%.2f GB reclaimed)" % size)

    print("\n  Compare against the current extraction: Troponin T covered")
    print("  20,441 of 203,016 stays.  If Troponin I or CK-MB add materially")
    print("  to that, wire lab_values_expanded.parquet into preprocess.py")
    print("  (engineer_labs) and rerun the pipeline.")


if __name__ == "__main__":
    main()
