# -*- coding: utf-8 -*-
"""Figures for the conference paper.

Four single-column figures, black and white throughout, in the plain line-art
style used by IEEE conference papers. Nothing spans the page, so the document
stays one continuous two-column section.

  fig1_architecture.png  layered system architecture
  fig4_results.png       deferral policy and disclosure horizon
  fig3_training.png      C2 validation AUROC and training loss over 40 epochs
  fig2_per_class.png     C1 per-pathology discrimination and sensitivity

The training curves in fig3 are the values logged in the component's own
history files for seeds 0, 1 and 2.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import json
import re
import numpy as np

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


OUT = r"c:\Users\94775\Desktop\box\R26-IT-083\Research Paper\figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "axes.edgecolor": "black",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "savefig.dpi": 400,
})

K = "black"
BLUE = "#1f77b4"     # matplotlib default cycle, as the components plot it
RED = "#d62728"
GREEN = "#2ca02c"
FAINT = "#9ecae1"


def grid(ax):
    ax.grid(True, color="0.85", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, lw=0.8):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=K,
                           facecolor="white", zorder=2))


def down(ax, x, y1, y2, lw=0.7):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                 mutation_scale=5.5, linewidth=lw, color=K,
                                 shrinkA=0, shrinkB=0, zorder=3))


# --------------------------------------------------------------------------
def figure1():
    """Layered block diagram: inputs, models, abstention rules, service."""
    fig, ax = plt.subplots(figsize=(3.4, 2.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lane = [
        ("Chest\nradiograph", "ConvNeXt-B", "per-group\nthreshold,\ndeferral"),
        ("12-lead\nECG", "1-D ResNet", "quality gate,\nconformal\nzones"),
        ("Echo\nvideo", "R(2+1)D-18", "conformal\ninterval,\nvariance"),
        ("ED triage\nrecord", "LightGBM", "horizon H,\nconstrained\nreferral"),
    ]
    n = len(lane)
    gap = 0.018
    w = (1.0 - (n - 1) * gap) / n
    xs = [i * (w + gap) for i in range(n)]

    rows = [(0.845, 0.135, 0, 5.6),
            (0.665, 0.100, 1, 5.8),
            (0.420, 0.150, 2, 5.4)]
    for y, h, idx, fs in rows:
        for i, x in enumerate(xs):
            box(ax, x, y, w, h)
            ax.text(x + w / 2, y + h / 2, lane[i][idx], ha="center",
                    va="center", fontsize=fs, linespacing=1.30, zorder=4)

    for x in xs:
        down(ax, x + w / 2, 0.843, 0.769)
        down(ax, x + w / 2, 0.663, 0.572)
        down(ax, x + w / 2, 0.418, 0.342)

    for y, lab in ((0.9125, "inputs"), (0.715, "models"),
                   (0.5125, "abstention")):
        ax.text(-0.020, y, lab, rotation=90, ha="center", va="center",
                fontsize=5.0)

    box(ax, 0, 0.235, 1.0, 0.105)
    ax.text(0.5, 0.2875, "Adapter layer  |  module sandbox", ha="center",
            va="center", fontsize=6.2, zorder=4)
    ax.text(0.5, 0.252, "each component's own frozen rule, run by its own code",
            ha="center", va="center", fontsize=5.1, zorder=4)
    down(ax, 0.5, 0.233, 0.175)

    box(ax, 0, 0.068, 1.0, 0.105, lw=1.3)
    ax.text(0.5, 0.1405, "Reliability envelope", ha="center", va="center",
            fontsize=6.2, fontweight="bold", zorder=4)
    ax.text(0.5, 0.093,
            "actionable | caution | deferred | withheld | unavailable",
            ha="center", va="center", fontsize=5.2, zorder=4)
    down(ax, 0.5, 0.066, 0.052)

    box(ax, 0.235, 0.0, 0.53, 0.050)
    ax.text(0.5, 0.025, "Clinical console", ha="center", va="center",
            fontsize=6.0, zorder=4)

    fig.subplots_adjust(left=0.040, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
    fig.subplots_adjust(left=0.145, right=0.985, top=0.845, bottom=0.30,
                        wspace=0.55)

    ax = axes[0]
    gaps, lo, hi = c1_deferral()
    err = [[g - l for g, l in zip(gaps, lo)], [h - g for g, h in zip(gaps, hi)]]
    xs = np.arange(3)
    bars = ax.bar(xs, gaps, width=0.6, color=[BLUE, BLUE, GREEN],
                  edgecolor=K, linewidth=0.6, zorder=2)
    bars[2].set_hatch("////")
    ax.errorbar(xs, gaps, yerr=err, fmt="none", ecolor=K, elinewidth=0.7,
                capsize=2, capthick=0.7, zorder=3)
    ax.axhline(0, color=K, linewidth=0.7, zorder=3)
    grid(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "equal\nrate", "per\ngroup"], fontsize=5.4)
    ax.set_ylabel("AP $-$ PA gap (points)", fontsize=5.6)
    ax.tick_params(axis="y", labelsize=5.4, length=2)
    ax.set_yticks([-2.5, 0, 2.5, 5.0, 7.5])
    ax.set_ylim(-4.5, 9.5)
    ax.set_title("(a)  deferral policy", fontsize=6.0, pad=3)

    ax = axes[1]
    H = [0, 6, 24]
    screen, ua, labs = c4_horizon(tuple(H))
    ax.plot(H, screen, marker="s", markersize=2.6, color=BLUE,
            linewidth=1.1, linestyle="-", zorder=3)
    ax.plot(H, ua, marker="o", markersize=2.6, color=RED,
            linewidth=1.1, linestyle="--", zorder=3)
    ax.plot(H, labs, marker="^", markersize=2.6, color=GREEN,
            linewidth=1.1, linestyle=":", zorder=3)
    grid(ax)
    ax.annotate("screen AUROC", (0.8, 0.985), fontsize=4.9)
    ax.annotate("UA recall", (13.0, 0.585), fontsize=4.9)
    ax.annotate("lab evidence", (0.8, 0.120), fontsize=4.9)
    ax.set_xticks(H)
    ax.set_xticklabels(["0", "6", "24"], fontsize=5.4)
    ax.set_xlabel("horizon $H$ (hours)", fontsize=5.6)
    ax.set_xlim(-1.5, 26)
    ax.set_ylim(-0.06, 1.18)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=5.4, length=2)
    ax.set_ylabel("score", fontsize=5.6)
    ax.set_title("(b)  disclosure horizon", fontsize=6.0, pad=3)

    fig.savefig(os.path.join(OUT, "fig4_results.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
#: read straight from the component's own training history files
AUROC, LOSS = c2_training()


def figure3():
    """Validation discrimination against training loss, three seeds."""
    fig, ax = plt.subplots(figsize=(3.4, 1.62))
    fig.subplots_adjust(left=0.135, right=0.865, top=0.90, bottom=0.235)
    ep = np.arange(1, 41)

    grid(ax)
    for a in AUROC:
        ax.plot(ep, a, color=FAINT, linewidth=0.55, zorder=2)
    ax.plot(ep, np.mean(AUROC, axis=0), color=BLUE, linewidth=1.2,
            label="validation macro-AUROC", zorder=4)
    best = int(np.argmax(np.mean(AUROC, axis=0))) + 1
    ax.axvline(best, color="0.35", linewidth=0.6, linestyle=":", zorder=3)
    ax.annotate("best epoch %d" % best, (best + 0.8, 0.9165), fontsize=4.9)
    ax.set_xlabel("epoch", fontsize=5.8)
    ax.set_ylabel("macro-AUROC", fontsize=5.8)
    ax.set_ylim(0.910, 0.945)
    ax.set_yticks([0.91, 0.92, 0.93, 0.94])
    ax.tick_params(labelsize=5.4, length=2)

    ax2 = ax.twinx()
    ax2.plot(ep, np.mean(LOSS, axis=0), color=RED, linewidth=1.1,
             linestyle="--", label="training loss", zorder=4)
    ax2.set_ylabel("training loss", fontsize=5.8)
    ax2.set_ylim(0.0, 0.13)
    ax2.set_yticks([0.0, 0.05, 0.10])
    ax2.tick_params(labelsize=5.4, length=2)

    ax.annotate("validation macro-AUROC", (21.5, 0.9422), fontsize=4.9,
                color=BLUE)
    ax2.annotate("training loss", (25.5, 0.0555), fontsize=4.9, color=RED)
    ax.set_title("C2 training behaviour, three seeds", fontsize=6.2, pad=3)
    fig.savefig(os.path.join(OUT, "fig3_training.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def figure4():
    """C1 discrimination and sensitivity per finding, ordered by AUROC."""
    fig, ax = plt.subplots(figsize=(3.4, 1.72))
    fig.subplots_adjust(left=0.315, right=0.975, top=0.885, bottom=0.245)

    labels, auroc, sens = c1_per_finding()
    labels = [n[0].upper() + n[1:].lower() for n in labels]

    y = np.arange(len(labels))[::-1]
    grid(ax)
    ax.barh(y + 0.19, auroc, height=0.36, color=BLUE, edgecolor=K,
            linewidth=0.5, label="AUROC", zorder=2)
    ax.barh(y - 0.19, sens, height=0.36, color=RED, edgecolor=K,
            linewidth=0.5, label="sensitivity", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=5.4, length=2)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0.75, color="0.25", linewidth=0.7, linestyle=":", zorder=3)
    ax.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.085), columnspacing=1.6,
              handlelength=1.5, handletextpad=0.5)
    ax.set_title("C1 per-finding discrimination and sensitivity",
                 fontsize=6.2, pad=3)
    fig.savefig(os.path.join(OUT, "fig2_per_class.png"))
    plt.close(fig)


figure1()
figure2()
figure3()
figure4()
print("written to", OUT)
