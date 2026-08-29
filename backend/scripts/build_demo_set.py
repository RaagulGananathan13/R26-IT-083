"""
Build a curated demo set: held-out studies each component gets right.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It selects studies from each component's own held-out test split, runs them
through the live serving path, and keeps the ones where the prediction matches
the ground-truth label AND the component stands behind it. The result is a
folder you can drag into the console during a demonstration and know in advance
what will happen.

It is NOT a performance sample. Selecting the cases a model gets right and then
quoting the hit rate would be circular. Each component's real, unselected
test-set figures are written into the manifest and the README beside every
curated file, so the two can never be confused.

The point of curating is to remove the risk of a demo failing for reasons that
have nothing to do with the work — a corrupt file, an unlucky draw, a study
outside the label space — not to make the models look better than they are.

DATA GOVERNANCE
---------------
Three of the four modalities are credentialed. The files copied here are
therefore excluded from git by `.gitignore`; only the manifest and README are
tracked. Do not add this folder to an archive you send anywhere.

USAGE
-----
    python scripts/build_demo_set.py                 # all four
    python scripts/build_demo_set.py --only cxr echo
    python scripts/build_demo_set.py --per-cell 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

REPO_ROOT = BACKEND_DIR.parent
DEMO_DIR = REPO_ROOT / "demo"

#: Unselected test-set performance, so a reader can never mistake a curated
#: folder for a measured hit rate.
TRUE_PERFORMANCE = {
    "cxr": "Cardiomegaly AUROC 0.9189, sensitivity 92.3 %, specificity 74.0 % "
           "on the full n=4,722 test set.",
    "ecg": "Macro accuracy 0.864, macro recall 0.810 on the untouched test "
           "fold 10 (n=1,711).",
    "echo": "MAE 3.979 EF points, 73.0 % overall accuracy, min per-class recall "
            "0.723 on the untouched test split (n=1,277).",
    "triage": "Stage-1 AUROC 0.9560 with NPV 99.41 %; Stage-2 subtyping "
              "macro-F1 0.7448 on the patient-disjoint test fold.",
}

ACCEPTABLE = {"actionable", "caution"}


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------
def build_cxr(registry, per_cell: int, out_dir: Path) -> List[Dict[str, Any]]:
    """Correct, actionable radiographs covering AP/PA x positive/negative."""
    import pandas as pd

    adapter = registry.get("cxr")
    root = adapter.root
    if root is None:
        return []
    image_root = root.parent / "data" / "output" / "cardio_image_384"
    manifest_path = root / "training_manifest" / "manifest_test.csv"
    if not image_root.is_dir() or not manifest_path.exists():
        log("  [cxr] credentialed images not present; skipping")
        return []

    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest = manifest[[(image_root / p).exists() for p in manifest.image_path]]
    adapter.ensure_loaded()

    selected: List[Dict[str, Any]] = []
    for view in ("PA", "AP"):
        for label in (1, 0):
            pool = manifest[
                (manifest["view"] == view) & (manifest["Cardiomegaly"] == label)
            ].sample(frac=1.0, random_state=20260819)
            kept = 0
            for _, row in pool.iterrows():
                if kept >= per_cell:
                    break
                path = image_root / row["image_path"]
                try:
                    envelope = adapter.analyze(
                        image_bytes=path.read_bytes(), view=view, filename=path.name)
                except Exception:                        # noqa: BLE001
                    continue
                finding = next(
                    (f for f in envelope.findings if f.name == "Cardiomegaly"), None)
                if finding is None or finding.present != bool(label):
                    continue
                if envelope.reliability.actionability.value not in ACCEPTABLE:
                    continue

                name = "%s_%s_%02d.png" % (
                    view, "cardiomegaly" if label else "normal", kept + 1)
                shutil.copy2(path, out_dir / name)
                selected.append({
                    "file": name,
                    "ground_truth": "Cardiomegaly" if label else "No cardiomegaly",
                    "projection": view,
                    "predicted": envelope.headline,
                    "probability": round(finding.probability or 0.0, 4),
                    "threshold": round(finding.threshold or 0.0, 4),
                    "verdict": envelope.reliability.actionability.value,
                    "how_to_use": "Upload the file, then set Projection to the "
                                  "value in its name (PA or AP). The projection "
                                  "selects the operating point, so it changes the "
                                  "result.",
                    "source_id": str(row["dicom_id"]),
                })
                kept += 1
                log("    %s  p=%.3f  %s" % (name, finding.probability or 0,
                                            envelope.reliability.actionability.value))
    return selected


# --------------------------------------------------------------------------
def build_ecg(registry, per_class: int, out_dir: Path) -> List[Dict[str, Any]]:
    """Records whose true superclass the model rules IN and stands behind."""
    import pandas as pd

    adapter = registry.get("ecg")
    root = adapter.root
    if root is None:
        return []
    csv_path = root / "csv" / "test.csv"
    signal_root = root / "data" / "raw_signals"
    if not csv_path.exists() or not signal_root.is_dir():
        log("  [ecg] test.csv or raw signals not present; skipping")
        return []

    frame = pd.read_csv(csv_path)
    adapter.ensure_loaded()
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    selected: List[Dict[str, Any]] = []
    for class_name in classes:
        column = "label_%s" % class_name
        if column not in frame.columns:
            continue
        pool = frame[frame[column] == 1].sample(frac=1.0, random_state=20260819)
        kept = 0
        for _, row in pool.iterrows():
            if kept >= per_class:
                break
            stem = str(row["filename_hr"])
            dat, hea = signal_root / (stem + ".dat"), signal_root / (stem + ".hea")
            if not (dat.exists() and hea.exists()):
                continue      # the bundled dataset is partial: 3,001 of 21,799
            try:
                envelope = adapter.analyze(
                    dat_bytes=dat.read_bytes(), hea_bytes=hea.read_bytes(),
                    record_name=Path(stem).name, with_xai=False)
            except Exception:                            # noqa: BLE001
                continue
            zones = (envelope.raw or {}).get("zones") or {}
            if zones.get(class_name) != "rule_in":
                continue
            if envelope.reliability.actionability.value not in ACCEPTABLE:
                continue

            base = "%s_%s" % (class_name, Path(stem).name)
            shutil.copy2(dat, out_dir / (base + ".dat"))
            shutil.copy2(hea, out_dir / (base + ".hea"))
            selected.append({
                "file": base + ".dat  +  " + base + ".hea",
                "ground_truth": class_name,
                "predicted": envelope.headline,
                "zone": "rule_in",
                "verdict": envelope.reliability.actionability.value,
                "how_to_use": "Upload BOTH files; they must share a base name.",
                "source_id": str(row["ecg_id"]),
            })
            kept += 1
            log("    %s  ruled in  %s" % (base, envelope.reliability.actionability.value))
    return selected


# --------------------------------------------------------------------------
def build_echo(registry, per_class: int, out_dir: Path) -> List[Dict[str, Any]]:
    """Cached clips whose true severity grade the ensemble reproduces."""
    import pandas as pd

    adapter = registry.get("echo")
    root = adapter.root
    if root is None:
        return []
    manifest_path = root / "preprocessing" / "artifacts" / "manifest.csv"
    cache = root / "preprocessing" / "cache" / "videos"
    if not manifest_path.exists() or not cache.is_dir():
        log("  [echo] manifest or clip cache not present; skipping")
        return []

    frame = pd.read_csv(manifest_path)
    frame = frame[frame["Split"].astype(str).str.upper() == "TEST"]
    adapter.ensure_loaded()
    names = ["Severe(<30)", "Moderate(30-40)", "Mild(40-55)", "Normal(>=55)"]
    labels = ["severe", "moderate", "mild", "normal"]

    selected: List[Dict[str, Any]] = []
    for index, class_name in enumerate(names):
        pool = frame[frame["ef_class"] == index].sample(frac=1.0, random_state=20260819)
        kept = 0
        for _, row in pool.iterrows():
            if kept >= per_class:
                break
            clip = cache / ("%s.npy" % row["FileName"])
            if not clip.exists():
                continue
            try:
                envelope = adapter.analyze(
                    video_bytes=clip.read_bytes(), filename=clip.name)
            except Exception:                            # noqa: BLE001
                continue
            raw = envelope.raw or {}
            if raw.get("severity_class") != class_name:
                continue
            verdict = envelope.reliability.actionability.value

            # Moderate is not excluded when it defers, and that is the point.
            # The band is 30-40 EF, so every Moderate study is within a few
            # points of a boundary by construction and a boundary-proximity
            # rule defers all of them -- measured here at 8 of 8, with 7 graded
            # correctly. That is Component 03's own finding about selective
            # prediction, and a demo set that quietly dropped the class would
            # hide the component's most interesting result.
            note = None
            if verdict not in ACCEPTABLE:
                if class_name != "Moderate(30-40)":
                    continue
                note = ("Graded correctly but DEFERRED. Moderate spans only "
                        "30-40 EF, so every case in this class sits near a "
                        "boundary and the conformal interval straddles it. "
                        "Expect a referral, not a call -- this is the component "
                        "behaving as designed.")

            name = "%s_%02d_%s.npy" % (labels[index], kept + 1, row["FileName"])
            shutil.copy2(clip, out_dir / name)
            selected.append({
                "file": name,
                "ground_truth": "%s (true EF %.1f %%)" % (class_name, float(row["EF"])),
                "predicted": envelope.headline,
                "predicted_ef": raw.get("ef_calibrated"),
                "interval_95": raw.get("ef_interval_95"),
                "verdict": verdict,
                "note": note,
                "how_to_use": "Upload directly; .npy is a cached clip array.",
                "source_id": str(row["FileName"]),
            })
            kept += 1
            log("    %s  EF %.1f vs true %.1f  %s" % (
                name, float(raw.get("ef_calibrated", 0)), float(row["EF"]), verdict))
    return selected


# --------------------------------------------------------------------------
def build_triage(registry, out_dir: Path) -> List[Dict[str, Any]]:
    """The synthetic ED records, verified against the live model."""
    source = BACKEND_DIR / "samples" / "triage"
    if not source.is_dir():
        log("  [triage] samples not generated; run make_sample_triage_pdfs.py")
        return []

    from cvxai.schemas.triage import TriageRequest
    from cvxai.services.pdf_triage import extract_triage_record

    adapter = registry.get("triage")
    adapter.ensure_loaded()

    expected = {
        "sample_01_stemi.pdf": "STEMI",
        "sample_02_nstemi.pdf": "NSTEMI",
        "sample_03_unstable_angina.pdf": "deferral (UA is the hardest class)",
        "sample_04_non_cardiac.pdf": "No_ACS",
        "sample_05_sparse.pdf": "deferral (sparse record)",
    }

    selected: List[Dict[str, Any]] = []
    for pdf in sorted(source.glob("*.pdf")):
        extraction = extract_triage_record(pdf.read_bytes())
        envelope = adapter.analyze(
            request=TriageRequest.model_validate(extraction.fields))
        raw = envelope.raw or {}
        shutil.copy2(pdf, out_dir / pdf.name)
        selected.append({
            "file": pdf.name,
            "ground_truth": expected.get(pdf.name, "—"),
            "predicted": raw.get("prediction"),
            "p_acs": round(float(raw.get("p_acs", 0.0)), 4),
            "verdict": envelope.reliability.actionability.value,
            "extraction_gaps": extraction.not_found,
            "how_to_use": "Upload on the Triage console's PDF tab, or load it "
                          "from the sample list there.",
            "synthetic": True,
        })
        log("    %-32s -> %-7s %s" % (pdf.name, raw.get("prediction"),
                                      envelope.reliability.actionability.value))
    return selected


# --------------------------------------------------------------------------
BUILDERS = {
    "cxr": ("01_chest_xray", build_cxr),
    "ecg": ("02_ecg", build_ecg),
    "echo": ("03_echocardiogram", build_echo),
    "triage": ("04_ed_triage", build_triage),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(BUILDERS),
                        default=list(BUILDERS))
    parser.add_argument("--per-cell", type=int, default=2,
                        help="studies per class / per projection-label cell")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from cvxai.core.logging import configure_logging
    from cvxai.core.registry import get_registry
    from cvxai.settings import get_settings

    configure_logging("ERROR")                   # component chatter is not useful here
    registry = get_registry(get_settings())
    out_root = Path(args.out) if args.out else DEMO_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    manifest: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what_this_is": (
            "Curated studies from each component's held-out test split, selected "
            "because the component predicts them correctly and stands behind the "
            "result. Chosen so a demonstration does not fail for reasons unrelated "
            "to the work."),
        "what_this_is_not": (
            "A performance sample. These cases were selected BY the models' own "
            "correctness, so their hit rate here is 100 % by construction and means "
            "nothing. The unselected test-set figures are in `true_performance`."),
        "true_performance": TRUE_PERFORMANCE,
        "components": {},
    }

    for component_id in args.only:
        folder, builder = BUILDERS[component_id]
        out_dir = out_root / folder
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        log("\n[%s] selecting into %s" % (component_id, folder))
        try:
            if component_id == "triage":
                entries = builder(registry, out_dir)
            else:
                entries = builder(registry, args.per_cell, out_dir)
        except Exception as exc:                         # noqa: BLE001
            log("  [%s] failed: %s" % (component_id, exc))
            entries = []
        manifest["components"][component_id] = {
            "folder": folder,
            "count": len(entries),
            "true_performance": TRUE_PERFORMANCE[component_id],
            "studies": entries,
        }
        log("  [%s] %d selected" % (component_id, len(entries)))

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(out_root, manifest)

    total = sum(c["count"] for c in manifest["components"].values())
    log("\n%d studies written to %s in %.1f min"
        % (total, out_root, (time.perf_counter() - started) / 60))
    return 0


def write_readme(out_root: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Demo set — R26-IT-083",
        "",
        "Studies each component predicts correctly, ready to drag into the console.",
        "",
        "> ⚠️ **Credentialed data.** Components 01–03 draw on MIMIC-CXR, PTB-XL and",
        "> EchoNet-Dynamic, all governed by data use agreements. This folder is",
        "> excluded from git and must not be copied into an archive you send",
        "> anywhere. Only Component 04's PDFs are synthetic and freely shareable.",
        "",
        "## What this is",
        "",
        manifest["what_this_is"],
        "",
        "## What this is not",
        "",
        manifest["what_this_is_not"],
        "",
    ]
    for component_id, block in manifest["components"].items():
        if block["count"] == 0:
            continue
        lines += [
            "---",
            "",
            "## `%s`" % block["folder"],
            "",
            "*Unselected test-set performance: %s*" % block["true_performance"],
            "",
            "| File | Ground truth | Predicted | Verdict |",
            "|---|---|---|---|",
        ]
        for study in block["studies"]:
            lines.append("| `%s` | %s | %s | %s |" % (
                study["file"], study["ground_truth"],
                str(study.get("predicted", ""))[:64], study["verdict"]))
        noted = [s for s in block["studies"] if s.get("note")]
        if noted:
            lines.append("")
            for study in noted:
                lines.append("- **`%s`** — %s" % (study["file"], study["note"]))
        lines += ["", "**How to use.** %s" % block["studies"][0]["how_to_use"], ""]

    lines += [
        "---",
        "",
        "## Regenerating",
        "",
        "```bash",
        "cd backend",
        "python scripts/build_demo_set.py",
        "```",
        "",
        "Selection re-runs against the live models, so a study that stops being",
        "predicted correctly drops out rather than silently going stale.",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
