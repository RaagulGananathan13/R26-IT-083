"""
COMPONENT_01 · Extract cardiomegaly-correct cases
=================================================

The two sets that matter for the primary target:

    cardiomegaly_present/  cardiomegaly was there AND the report said so
    cardiomegaly_absent/   cardiomegaly was absent AND the report did not claim it

Both filtered to reports of good overall quality (>= 7 of 8 findings agreeing with
the radiologist), so these are cases where the whole report is sound, not just the
one finding. Co-pathologies are allowed and expected -- a correct report of
"cardiomegaly with pleural effusion" belongs here.

ONE DISTINCTION THAT MATTERS FOR THE ABSENT SET
-----------------------------------------------
A report can fail to claim cardiomegaly in two very different ways:

    EXPLICIT  "the cardiomediastinal silhouette is normal"  <- states the negative
    SILENT    never mentions the heart at all                <- says nothing

Both score identically on every automatic metric. Only the first is a usable
demonstration -- a report that simply omits the heart is not evidence the model
assessed it. Explicit negatives are therefore copied first and flagged in
manifest.csv, so a silent report is never mistaken for a confident one.

READ-ONLY on the dataset and manifests. Adds to review_cases/ without touching
the folders already there.

Run:  python extract_cardiomegaly_cases.py
"""
from __future__ import annotations
import re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
SRC_IMG = HERE.parent / "data" / "output" / "cardio_image_384"
MANIFEST = HERE / "training_manifest" / "manifest_test.csv"
S12 = HERE / "reports" / "stage12"
PROBS = HERE / "reports" / "stage6" / "cache" / "probs_test.npy"
OUT = HERE / "review_cases"

sys.path.insert(0, str(HERE))
from build_review import findings, PATHOLOGIES, KW      # noqa: E402

MIN_MATCH = 7          # of 8 findings must agree with the radiologist
CAP = 400              # images copied per folder; manifest.csv lists every case
HEART_RE = re.compile(KW["Cardiomegaly"], re.I)


def mentions_heart(text: str) -> bool:
    """Does the report refer to the cardiac silhouette at all, either way?"""
    # Deliberately broad. A first version matched only "cardiac silhouette" and
    # scored "the cardiac AND MEDIASTINAL silhouettes are normal" as silent --
    # undercounting explicit negatives by a wide margin.
    return bool(HEART_RE.search(text or "")) or bool(re.search(
        r"(cardiomediastin"
        r"|cardiac(\s+\w+){0,3}\s+(silhouette|contour|shadow|size)"
        r"|heart\s+(size|is|appears|remains|and)"
        r"|(silhouette|contour)s?\s+(are|is)\s+\w*\s*normal"
        r"|mediastinal\s+silhouette)", text or "", re.I))


def main():
    o = OUT.resolve()
    for prot in (SRC_IMG, HERE / "training_manifest", HERE.parent / "data"):
        p = prot.resolve()
        assert o != p and p not in o.parents, "output would land inside " + str(p)

    te = pd.read_csv(MANIFEST, low_memory=False)
    REF = (S12 / "references_test.txt").read_text(encoding="utf-8").split("\n")
    GEN = (S12 / "reports_stage11_test.txt").read_text(encoding="utf-8").split("\n")
    PR = np.load(PROBS)
    n = min(len(te), len(REF), len(GEN), len(PR))
    norm = lambda s: " ".join(str(s).split())
    assert all(norm(REF[i]) == norm(te.report.iloc[i]) for i in range(n)), "misalignment"
    print("  alignment verified on all %d rows" % n)

    fr = [findings(REF[i]) for i in range(n)]
    fg = [findings(GEN[i]) for i in range(n)]
    agree = np.array([sum(fr[i][k] == fg[i][k] for k in PATHOLOGIES) for i in range(n)])
    c_true = te["Cardiomegaly"].to_numpy()[:n]
    c_gen = np.array([fg[i]["Cardiomegaly"] for i in range(n)])
    heart = np.array([mentions_heart(GEN[i]) for i in range(n)])
    n_co = np.array([sum(fr[i][k] for k in PATHOLOGIES if k != "Cardiomegaly")
                     for i in range(n)])

    good = agree >= MIN_MATCH
    sets = {
        "cardiomegaly_present": (good & (c_true == 1) & (c_gen == 1),
                                 lambda i: (-agree[i], -PR[i][0])),      # confident first
        "cardiomegaly_absent":  (good & (c_true == 0) & (c_gen == 0),
                                 lambda i: (-agree[i], -int(heart[i]), PR[i][0])),
    }

    print()
    for name, (mask, key) in sets.items():
        idx = sorted(np.where(mask)[0].tolist(), key=key)
        d = OUT / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        rows, copied = [], 0
        for rank, i in enumerate(idx):
            src = SRC_IMG / te.image_path.iloc[i]
            take = rank < CAP and src.exists()
            if take:
                shutil.copy2(src, d / ("%s.png" % te.dicom_id.iloc[i]))
                copied += 1
            rows.append(dict(
                rank=rank, case_index=int(i), dicom_id=te.dicom_id.iloc[i],
                view=te.view.iloc[i], image_copied=bool(take),
                findings_matched=int(agree[i]),
                cardiomegaly_true=int(c_true[i]),
                report_states_cardiomegaly=int(c_gen[i]),
                report_mentions_heart=bool(heart[i]),
                classifier_p_cardiomegaly=round(float(PR[i][0]), 4),
                co_pathologies_in_real_report=int(n_co[i]),
                **{"true_" + k: int(te[k].iloc[i]) for k in PATHOLOGIES},
                real_report=norm(REF[i]), generated_report=norm(GEN[i])))
        df = pd.DataFrame(rows)
        df.to_csv(d / "manifest.csv", index=False)
        expl = int(df.report_mentions_heart.sum())
        print("  %-24s %4d cases (%d with 8/8), %3d images copied" %
              (name, len(idx), int((agree[idx] == 8).sum()), copied))
        print("       report explicitly mentions the heart: %d of %d (%.0f%%)" %
              (expl, len(idx), 100 * expl / max(len(idx), 1)))
        print("       AP %d / PA %d   mean co-pathologies %.2f" %
              (int((te.view.iloc[idx] == "AP").sum()),
               int((te.view.iloc[idx] == "PA").sum()), df.co_pathologies_in_real_report.mean()))
        print()

    (OUT / "CARDIOMEGALY_README.md").write_text("""# Cardiomegaly Cases

Cases where the generated report got **cardiomegaly right** and the report as a
whole is sound (**>= %d of 8 findings agree** with the radiologist).

| Folder | Meaning |
|---|---|
| `cardiomegaly_present/` | cardiomegaly was present **and** the report stated it |
| `cardiomegaly_absent/` | cardiomegaly was absent **and** the report did not claim it |

Co-pathologies are allowed. A correct report of "cardiomegaly with pleural
effusion" belongs in `cardiomegaly_present/` — the column
`co_pathologies_in_real_report` tells you how many other findings the radiologist
noted.

## ⚠️ Read this before picking demo cases

`report_mentions_heart` distinguishes two very different things in the **absent**
set:

- **`True`** — the report explicitly assessed the heart and found it normal
  (*"the cardiomediastinal silhouette is within normal limits"*). **Use these.**
- **`False`** — the report never mentions the heart at all. It scores identically
  on every automatic metric, but it is **not** evidence the model assessed
  anything. Do not present these as correct negatives.

Cases are sorted best-first: highest `findings_matched`, then explicit mentions,
then classifier confidence. So `rank` 0 is the strongest case in each folder.

## Columns

`rank` · `case_index` · `dicom_id` · `view` · `findings_matched` ·
`cardiomegaly_true` · `report_states_cardiomegaly` · `report_mentions_heart` ·
`classifier_p_cardiomegaly` · `co_pathologies_in_real_report` · `true_*` labels ·
`real_report` · `generated_report`

`case_index` is the row in `manifest_test.csv`.

## ⚠️ MIMIC-CXR — PhysioNet Data Use Agreement

Credentialed clinical images. Inspect locally and show a few in a presentation.
**Never** commit to GitHub or share with anyone without credentials.
""" % MIN_MATCH, encoding="utf-8")

    mb = sum(f.stat().st_size for d in ("cardiomegaly_present", "cardiomegaly_absent")
             for f in (OUT / d).glob("*.png")) / 1e6
    print("  wrote %s  (+%.0f MB)" % (OUT, mb))
    print("  dataset and manifests untouched")


if __name__ == "__main__":
    main()
