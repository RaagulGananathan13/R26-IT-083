# -*- coding: utf-8 -*-
"""One metric that means the same thing in all four modalities.

Unsafe answer rate (UAR) = P(the system answers AND the answer is wrong)
                         = coverage * (1 - accuracy among answered)

A model that never abstains has UAR = its error rate. A model that abstains
perfectly has UAR = 0 at the cost of coverage. Reported beside coverage, it
prices abstention instead of hiding it, which accuracy-on-the-answered-subset
does not.

Every input is read from the components' own artifacts.
"""
import json
import os

ROOT = r"c:\Users\94775\Desktop\box\R26-IT-083"


def p(*a):
    q = os.path.join(ROOT, *a)
    if not os.path.exists(q):
        raise SystemExit("missing: " + q)
    return q


def uar(coverage, acc_answered):
    """coverage and accuracy both as fractions in [0, 1]."""
    return coverage * (1.0 - acc_answered)


out = {}

# ---------------------------------------------------------------- C1 ------
rows = json.load(open(p("Component_01", "Component_01", "reports", "stage13",
                        "summary.json"), encoding="utf-8"))["rows"]


def arm(name, cov=None):
    hit = [r for r in rows if r["arm"] == name
           and (cov is None or abs(r["coverage_target"] - cov) < 1e-9)]
    assert len(hit) == 1, (name, cov, len(hit))
    return hit[0]


a_none = arm("A none", 1.0)
a_glob = arm("C global", 0.8)
a_cond = arm("D conditional", 0.8)
out["C1"] = {
    "none":        (a_none["coverage"] / 100.0, a_none["accuracy"] / 100.0, a_none["gap"]),
    "uniform":     (a_glob["coverage"] / 100.0, a_glob["accuracy"] / 100.0, a_glob["gap"]),
    "conditional": (a_cond["coverage"] / 100.0, a_cond["accuracy"] / 100.0, a_cond["gap"]),
}

# ---------------------------------------------------------------- C3 ------
d3 = json.load(open(p("Component_03", "Dilukshan", "training", "outputs",
                      "selective_report.json"), encoding="utf-8"))
full, sel = d3["full_coverage"], d3["selective"]
n_tot, n_def = sel["n_total"], sel["n_deferred"]
n_cov = sel["n_covered"]
acc_all = full["overall_acc"]
acc_def = sel["deferred_accuracy"]
# correct answers among the covered subset, recovered from the two known rates
corr_cov = acc_all * n_tot - acc_def * n_def
acc_cov = corr_cov / n_cov
out["C3"] = {
    "none":       (1.0, acc_all, full["min_class_recall"]),
    "selective":  (sel["coverage"], acc_cov, sel["min_class_recall"]),
    "_deferred_accuracy": acc_def,
}

# ---------------------------------------------------------------- C4 ------
s2 = json.load(open(p("Component_04", "artifacts", "reports",
                      "stage2_metrics_H24.json"), encoding="utf-8"))["test"]
sel4 = json.load(open(p("Component_04", "artifacts", "reports",
                        "selective_H24.json"), encoding="utf-8"))
band = sel4["Stage 2 \u2014 subtyping_f1"]
out["C4"] = {
    "none":      (1.0, s2["accuracy"], s2["min_recall"]),
    "selective": (band["coverage"], band["test"]["accuracy"],
                  band["test"]["min_recall"]),
    "_recall_floor_attainable":
        sel4["Stage 2 \u2014 subtyping_recall"]["attainable"],
}

# ---------------------------------------------------------------- report --
print("component  arm           coverage   acc|answered   UAR      worst-class/gap")
print("-" * 78)
for comp in ("C1", "C3", "C4"):
    for k, v in out[comp].items():
        if k.startswith("_"):
            continue
        cov, acc, extra = v
        print("  %-8s %-13s %7.2f%%   %7.2f%%   %6.2f%%   %8.4f"
              % (comp, k, 100 * cov, 100 * acc, 100 * uar(cov, acc), extra))
    print()

print("C3 accuracy on the DEFERRED cases : %.4f  (vs %.4f overall)"
      % (out["C3"]["_deferred_accuracy"], out["C3"]["none"][1]))
print("C4 0.75 recall floor attainable   :", out["C4"]["_recall_floor_attainable"])
print("\nC3 min-class recall full->selective: %.4f -> %.4f"
      % (out["C3"]["none"][2], out["C3"]["selective"][2]))
