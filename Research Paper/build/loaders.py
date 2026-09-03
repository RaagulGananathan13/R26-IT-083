# -*- coding: utf-8 -*-
"""Reads every plotted value straight out of the components' own output files.

Nothing in the figures is typed in by hand. If a source file moves or its
contents change, these loaders raise instead of drawing a stale picture.
"""
import json
import os
import re

ROOT = r"c:\Users\94775\Desktop\box\R26-IT-083"


def _p(*parts):
    q = os.path.join(ROOT, *parts)
    if not os.path.exists(q):
        raise SystemExit("missing source artifact: " + q)
    return q


# ---------------------------------------------------------------- C2 training
def c2_training():
    """Per-epoch validation macro-AUROC and training loss, seeds 0/1/2."""
    auroc, loss = [], []
    for s in (0, 1, 2):
        h = json.load(open(_p("Component_02", "Component_02", "checkpoints",
                              "resnet", "history_seed%d.json" % s),
                           encoding="utf-8"))["history"]
        if len(h) != 40:
            raise SystemExit("seed %d: expected 40 epochs, found %d" % (s, len(h)))
        auroc.append([e["macro_auroc"] for e in h])
        loss.append([e["train_loss"] for e in h])
    return auroc, loss


# ---------------------------------------------------------------- C1 deferral
def c1_deferral(coverage_target=0.8):
    """AP-PA accuracy gap with bootstrap CI, for the three deferral arms."""
    rows = json.load(open(_p("Component_01", "Component_01", "reports",
                             "stage13", "summary.json"),
                          encoding="utf-8"))["rows"]

    def pick(arm, cov=None):
        hit = [r for r in rows if r["arm"] == arm
               and (cov is None or abs(r["coverage_target"] - cov) < 1e-9)]
        if len(hit) != 1:
            raise SystemExit("stage13: %d rows for arm %r at coverage %r"
                             % (len(hit), arm, cov))
        return hit[0]

    picked = [pick("A none", 1.0),
              pick("C global", coverage_target),
              pick("D conditional", coverage_target)]
    return ([r["gap"] for r in picked],
            [r["gap_lo"] for r in picked],
            [r["gap_hi"] for r in picked])


# ----------------------------------------------------------------- C4 horizon
def c4_horizon(horizons=(0, 6, 24)):
    """Screening AUROC, unstable-angina recall, laboratory SHAP share."""
    rep = (ROOT, "Component_04", "artifacts", "reports")
    auroc, ua, labs = [], [], []
    for h in horizons:
        s1 = json.load(open(_p(*rep[1:], "stage1_metrics_H%d.json" % h),
                            encoding="utf-8"))["test"]
        s2 = json.load(open(_p(*rep[1:], "stage2_metrics_H%d.json" % h),
                            encoding="utf-8"))["test"]
        ex = json.load(open(_p(*rep[1:], "explainability_H%d.json" % h),
                            encoding="utf-8"))
        auroc.append(float(s1["auroc"]))
        ua.append(float(s2["per_class"]["UA"]["recall"]))
        labs.append(float(ex["stage1_modality"]["labs"]))
    return auroc, ua, labs


# --------------------------------------------------------------- C1 per class
ROW = re.compile(r"^\|\s*\*{0,2}([A-Za-z ]+?)\*{0,2}\s*\|(.+)\|\s*$")


def c1_per_finding():
    """Per-pathology AUROC and sensitivity from the component's results table."""
    txt = open(_p("Component_01", "Component_01", "RESULTS.md"),
               encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(txt)
                 if l.startswith("| Pathology |") and "AUROC" in l)
    out = []
    for line in txt[start + 2:]:
        m = ROW.match(line)
        if not m:
            break
        name = m.group(1).strip()
        if name.upper() == "MEAN":
            break
        cells = [c.strip().replace("*", "") for c in m.group(2).split("|")]
        out.append((name, float(cells[-1]), float(cells[6]) / 100.0))
    if len(out) != 8:
        raise SystemExit("RESULTS.md: expected 8 pathologies, parsed %d" % len(out))
    out.sort(key=lambda r: -r[1])
    return ([r[0] for r in out], [r[1] for r in out], [r[2] for r in out])


if __name__ == "__main__":
    a, l = c2_training()
    print("C2 training   : %d seeds x %d epochs, AUROC[0][0]=%.4f loss[0][0]=%.4f"
          % (len(a), len(a[0]), a[0][0], l[0][0]))
    print("C1 deferral   :", [round(x, 4) for x in c1_deferral()[0]])
    ah, ua, lb = c4_horizon()
    print("C4 horizon    :", [round(x, 4) for x in ah],
          [round(x, 4) for x in ua], [round(x, 4) for x in lb])
    n, au, se = c1_per_finding()
    print("C1 per finding:", list(zip(n, [round(x, 4) for x in au],
                                      [round(x, 3) for x in se])))


# ------------------------------------------------- framework-level metric
def unsafe_answer_rates():
    """(coverage, accuracy-among-answered) per arm, straight from artifacts.

    Unsafe answer rate U = coverage * (1 - accuracy), the probability that the
    system both answers and is wrong.
    """
    out = {}

    rows = json.load(open(_p("Component_01", "Component_01", "reports",
                             "stage13", "summary.json"),
                          encoding="utf-8"))["rows"]

    def arm(name, cov):
        h = [r for r in rows if r["arm"] == name
             and abs(r["coverage_target"] - cov) < 1e-9]
        if len(h) != 1:
            raise SystemExit("stage13: %d rows for %r" % (len(h), name))
        return h[0]["coverage"] / 100.0, h[0]["accuracy"] / 100.0

    out["C1"] = [("answer all", arm("A none", 1.0)),
                 ("uniform", arm("C global", 0.8)),
                 ("per group", arm("D conditional", 0.8))]

    d3 = json.load(open(_p("Component_03", "Dilukshan", "training", "outputs",
                           "selective_report.json"), encoding="utf-8"))
    f, s = d3["full_coverage"], d3["selective"]
    acc_cov = (f["overall_acc"] * s["n_total"]
               - s["deferred_accuracy"] * s["n_deferred"]) / s["n_covered"]
    out["C3"] = [("answer all", (1.0, f["overall_acc"])),
                 ("selective", (s["coverage"], acc_cov))]

    rep = ("Component_04", "artifacts", "reports")
    s2 = json.load(open(_p(*rep, "stage2_metrics_H24.json"),
                        encoding="utf-8"))["test"]
    band = json.load(open(_p(*rep, "selective_H24.json"),
                          encoding="utf-8"))["Stage 2 \u2014 subtyping_f1"]
    out["C4"] = [("answer all", (1.0, s2["accuracy"])),
                 ("selective", (band["coverage"], band["test"]["accuracy"]))]
    return out
