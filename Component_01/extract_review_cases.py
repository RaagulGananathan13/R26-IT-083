"""
COMPONENT_01 · Extract review cases into inspectable folders
============================================================

Copies chest X-rays into folders grouped by how the generated report compared
with the radiologist's, so cases can be inspected visually.

READ-ONLY on everything original. Nothing under data/ or training_manifest/ is
opened for writing; images are copied FROM the dataset, never moved or altered.
The script asserts this before doing anything.

Grouping is deliberate. "All 8 findings matched" is a weak criterion on its own
-- a report that mentions nothing scores 8/8 on 25.7% of cases, because the
reference mentions nothing either. Lumping those together with genuine successes
would produce a folder of normal chest X-rays labelled "the model got these
perfectly right". So the perfect matches are split by whether the radiologist
actually reported any pathology.

Filenames use the DICOM ID, never the source folder name. The dataset's
test/negative/ and test/positive/ folders refer to CARDIOMEGALY ONLY -- a film
in negative/ can still have a moderate pleural effusion, and often does.

Run:  python extract_review_cases.py
Out:  review_cases/
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
SRC_IMG = HERE.parent / "data" / "output" / "cardio_image_384"
MANIFEST = HERE / "training_manifest" / "manifest_test.csv"
S12 = HERE / "reports" / "stage12"
PROBS = HERE / "reports" / "stage6" / "cache" / "probs_test.npy"
OUT = HERE / "review_cases"

sys.path.insert(0, str(HERE))
from build_review import findings, PATHOLOGIES          # noqa: E402

# How many images to copy per group. Full membership is always written to
# manifest.csv even when the image itself is not copied -- perfect_normal is
# capped hard because ~1,200 normal films teach you nothing.
CAPS = {"perfect_with_findings": 400, "perfect_normal": 50,
        "cardiomegaly_missed": 400, "cardiomegaly_false_positive": 250,
        "worst": 250}


def guard():
    """Refuse to run if the output could land inside protected directories."""
    o = OUT.resolve()
    for prot in (SRC_IMG, HERE / "training_manifest", HERE.parent / "data"):
        p = prot.resolve()
        assert o != p and p not in o.parents, "output would land inside " + str(p)
    for f in (MANIFEST, PROBS, S12 / "references_test.txt",
              S12 / "reports_stage11_test.txt"):
        assert f.exists(), "missing input: " + str(f)
    assert SRC_IMG.exists(), "image dataset not found: " + str(SRC_IMG)


def main():
    guard()
    te = pd.read_csv(MANIFEST, low_memory=False)
    REF = (S12 / "references_test.txt").read_text(encoding="utf-8").split("\n")
    GEN = (S12 / "reports_stage11_test.txt").read_text(encoding="utf-8").split("\n")
    PR = np.load(PROBS)
    n = min(len(te), len(REF), len(GEN), len(PR))

    # Alignment is the one thing that could silently produce wrong images.
    norm = lambda s: " ".join(str(s).split())
    bad = [i for i in range(n) if norm(REF[i]) != norm(te.report.iloc[i])]
    assert not bad, "reference/manifest misalignment at rows %s" % bad[:5]
    print("  alignment verified on all %d rows" % n)

    fr = [findings(REF[i]) for i in range(n)]
    fg = [findings(GEN[i]) for i in range(n)]
    agree = np.array([sum(fr[i][k] == fg[i][k] for k in PATHOLOGIES) for i in range(n)])
    n_ref = np.array([sum(fr[i].values()) for i in range(n)])
    c_true = te["Cardiomegaly"].to_numpy()[:n]
    c_gen = np.array([fg[i]["Cardiomegaly"] for i in range(n)])

    groups = {
        "perfect_with_findings": (agree == 8) & (n_ref >= 1),
        "perfect_normal":        (agree == 8) & (n_ref == 0),
        "cardiomegaly_missed":   (c_true == 1) & (c_gen == 0),
        "cardiomegaly_false_positive": (c_true == 0) & (c_gen == 1),
        "worst":                 agree <= 5,
    }

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    summary = []
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        d = OUT / name
        d.mkdir()
        cap = CAPS[name]
        copy_idx = set(idx[:cap].tolist())
        rows, copied = [], 0
        for i in idx:
            src = SRC_IMG / te.image_path.iloc[i]
            did = te.dicom_id.iloc[i]
            take = i in copy_idx and src.exists()
            if take:
                shutil.copy2(src, d / (str(did) + ".png"))   # READ from dataset
                copied += 1
            rows.append(dict(
                case_index=int(i), dicom_id=did, view=te.view.iloc[i],
                image_copied=bool(take), findings_matched=int(agree[i]),
                findings_in_real_report=int(n_ref[i]),
                cardiomegaly_true=int(c_true[i]),
                cardiomegaly_in_report=int(c_gen[i]),
                **{"p_" + k: round(float(PR[i][j]), 4) for j, k in enumerate(PATHOLOGIES)},
                **{"true_" + k: int(te[k].iloc[i]) for k in PATHOLOGIES},
                real_report=norm(REF[i]), generated_report=norm(GEN[i])))
        pd.DataFrame(rows).to_csv(d / "manifest.csv", index=False)
        summary.append((name, len(idx), copied))
        print("  %-30s %5d cases, %4d images copied" % (name, len(idx), copied))

    (OUT / ".gitignore").write_text("*\n", encoding="utf-8")
    (OUT / "README.md").write_text("""# Review Cases

Chest X-rays grouped by how the generated report compared with the radiologist's.

## ⚠️ MIMIC-CXR — PhysioNet Data Use Agreement

These are credentialed clinical images.

- ✅ inspect locally, show a few in a presentation
- ❌ **never** commit to GitHub, even privately
- ❌ **never** upload to a shared drive or send to anyone without credentials

A `.gitignore` excluding everything is included in this folder.

## Folders

| Folder | Cases | Images | What it contains |
|---|---|---|---|
""" + "\n".join("| `%s` | %d | %d | |" % (a, b, c) for a, b, c in summary) + """

- **perfect_with_findings** — every finding matched **and** the radiologist reported
  real pathology. The genuinely successful cases.
- **perfect_normal** — matched because neither report mentioned anything. Correct,
  but a report saying nothing scores 8/8 on 25.7% of cases. Capped at 50; not
  evidence of capability.
- **cardiomegaly_missed** — the heart was enlarged and the report did not say so.
  Clinically the most important failures.
- **cardiomegaly_false_positive** — over-called enlargement.
- **worst** — five or fewer findings matched. The real failure modes.

## manifest.csv

Each folder has one, listing **every** case in the group (including any whose image
was not copied due to the cap):

`case_index` · `dicom_id` · `view` · `findings_matched` · `cardiomegaly_true` ·
`cardiomegaly_in_report` · `p_*` classifier probabilities · `true_*` labels ·
`real_report` · `generated_report`

`case_index` is the row number in `manifest_test.csv` and matches the case numbers
in `reports/stage12/MANUAL_REVIEW.md`.

## Note on filenames

Files are named by **DICOM ID**. The source dataset's `test/negative/` and
`test/positive/` folders refer to **cardiomegaly only** — a film in `negative/`
can still have a moderate pleural effusion, and often does. Use `manifest.csv`
for ground truth, never the source folder name.
""", encoding="utf-8")

    total_mb = sum(f.stat().st_size for f in OUT.rglob("*.png")) / 1e6
    print("\n  wrote %s  (%.0f MB)" % (OUT, total_mb))
    print("  original dataset untouched (read-only copy)")


if __name__ == "__main__":
    main()
