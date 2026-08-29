"""
COMPONENT_01 · Manual review builder
====================================

Produces a human-readable side-by-side of generated vs real radiologist reports,
for qualitative assessment that no automatic metric captures.

Cases are NOT sampled at random. Random sampling on a corpus this templated shows
you fifty near-identical "lungs are clear" reports and teaches you nothing. This
stratifies deliberately toward the cases that carry information:

    BEST      the model got every finding right          -> what good looks like
    WORST     the model got the findings wrong           -> the real failure modes
    MISSED    cardiomegaly present, model said no        -> the clinically dangerous error
    FALSE +   cardiomegaly absent, model said yes        -> the over-calling behaviour
    TRUE +    cardiomegaly correctly detected            -> the working case

Run:  python build_review.py
Out:  reports/stage12/MANUAL_REVIEW.md
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
S12 = HERE / "reports" / "stage12"
GEN_F = S12 / "reports_stage11_test.txt"
REF_F = S12 / "references_test.txt"
OUT_F = S12 / "MANUAL_REVIEW.md"

PATHOLOGIES = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
               "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]
KW = {"Cardiomegaly": r"(cardiomegaly|cardiac enlargement|enlarged cardiac silhouette|heart.{0,20}enlarged)",
      "Edema": r"(pulmonary edema|interstitial edema|\bedema\b|vascular congestion)",
      "Pleural_Effusion": r"(pleural effusion|\beffusions?\b)", "Atelectasis": r"atelecta",
      "Consolidation": r"consolidat", "Lung_Opacity": r"(opacit|infiltrate)",
      "Pneumonia": r"pneumonia", "Pneumothorax": r"pneumothora"}
NEG = re.compile(r"\b(no|not|without|negative for|free of|absence of|absent)\b", re.I)


def findings(text: str) -> dict:
    """Which findings does this report assert? Sentence-scoped negation."""
    out, sents = {}, re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", text or ""))
    for k, pat in KW.items():
        kw, pos = re.compile(pat, re.I), 0
        for s in sents:
            for m in kw.finditer(s):
                if not NEG.search(s[:m.start()]):
                    pos = 1
                    break
            if pos:
                break
        out[k] = pos
    return out


def main():
    for f in (GEN_F, REF_F):
        if not f.exists():
            sys.exit("MISSING: %s\n\nDownload it from Drive:\n"
                     "  MyDrive/Component_01/reports/stage12/%s" % (f, f.name))

    GEN = GEN_F.read_text(encoding="utf-8").split("\n")
    REF = REF_F.read_text(encoding="utf-8").split("\n")
    te = pd.read_csv(HERE / "training_manifest" / "manifest_test.csv", low_memory=False)
    PR = np.load(HERE / "reports" / "stage6" / "cache" / "probs_test.npy")
    n = min(len(GEN), len(REF), len(te))
    GEN, REF = GEN[:n], REF[:n]
    print("  loaded %d report pairs" % n)

    fg = [findings(g) for g in GEN]
    fr = [findings(r) for r in REF]
    agree = np.array([sum(fg[i][k] == fr[i][k] for k in PATHOLOGIES) for i in range(n)])
    card_true = te["Cardiomegaly"].to_numpy()[:n]
    card_gen = np.array([fg[i]["Cardiomegaly"] for i in range(n)])

    rng = np.random.default_rng(0)

    def pick(mask, k, best=None):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        if best is True:
            idx = idx[np.argsort(-agree[idx])]
        elif best is False:
            idx = idx[np.argsort(agree[idx])]
        else:
            idx = rng.permutation(idx)
        return list(idx[:k])

    groups = [
        ("A · BEST — every finding matched", pick(agree == 8, 12),
         "What the system looks like when it works."),
        ("B · WORST — most findings wrong", pick(agree <= 5, 12, best=False),
         "The real failure modes. Read these most carefully."),
        ("C · MISSED cardiomegaly (false negative)", pick((card_true == 1) & (card_gen == 0), 10),
         "Clinically the dangerous error: the heart was enlarged, the report did not say so."),
        ("D · FALSE-POSITIVE cardiomegaly", pick((card_true == 0) & (card_gen == 1), 10),
         "Over-calling. Less dangerous, but erodes trust and causes unnecessary workup."),
        ("E · CORRECT cardiomegaly detection", pick((card_true == 1) & (card_gen == 1), 10),
         "The working case for the primary target."),
    ]

    L = ["# Manual Review — Generated vs Real Radiologist Reports",
         "",
         "*Component_01 · Stage 11 model · test set n = %d*" % n,
         "",
         "> Cases are **deliberately stratified, not random**. Random sampling on a corpus",
         "> this templated returns fifty near-identical \"lungs are clear\" reports. The",
         "> groups below are chosen to show you where the system works and — more",
         "> usefully — where it fails.",
         "",
         "## How to read each case",
         "",
         "| Field | Meaning |",
         "|---|---|",
         "| **REAL** | what the radiologist actually wrote |",
         "| **GENERATED** | what the model wrote |",
         "| **findings match** | of the 8 tracked pathologies, how many agree |",
         "| **classifier** | the image model's probability for each finding it flagged |",
         "| ✅ / ❌ | whether the generated report agrees with the real one on that finding |",
         "",
         "---", ""]

    for title, idxs, note in groups:
        L += ["## " + title, "", "*%s*" % note, ""]
        if not idxs:
            L += ["*(no cases in this group)*", "", "---", ""]
            continue
        for i in idxs:
            probs = {k: float(PR[i][j]) for j, k in enumerate(PATHOLOGIES)}
            flagged = sorted([k for k in PATHOLOGIES if probs[k] >= 0.5],
                             key=lambda k: -probs[k])
            L += ["### Case %d — %s view" % (i, te["view"].iloc[i]), "",
                  "**REAL (radiologist):**", "", "> " + REF[i].strip(), "",
                  "**GENERATED (model):**", "", "> " + GEN[i].strip(), "",
                  "| finding | real report | generated | |",
                  "|---|---|---|---|"]
            for k in PATHOLOGIES:
                if fr[i][k] == 0 and fg[i][k] == 0:
                    continue                       # neither mentions it, skip the noise
                L.append("| %s | %s | %s | %s |" % (
                    k.replace("_", " "),
                    "**yes**" if fr[i][k] else "no",
                    "**yes**" if fg[i][k] else "no",
                    "✅" if fr[i][k] == fg[i][k] else "❌"))
            L += ["",
                  "`findings match: %d/8`  ·  `classifier flagged: %s`" % (
                      agree[i],
                      ", ".join("%s %.2f" % (k.replace("_", " "), probs[k])
                                for k in flagged) or "nothing above 0.50"),
                  "", "---", ""]

    L += ["## Summary across all %d test cases" % n, "",
          "| | |", "|---|---|",
          "| all 8 findings matched | %d (%.1f%%) |" % ((agree == 8).sum(), (agree == 8).mean()*100),
          "| 7 or more matched | %d (%.1f%%) |" % ((agree >= 7).sum(), (agree >= 7).mean()*100),
          "| 5 or fewer matched | %d (%.1f%%) |" % ((agree <= 5).sum(), (agree <= 5).mean()*100),
          "| mean findings matched | %.2f / 8 |" % agree.mean(),
          "| cardiomegaly missed | %d |" % (((card_true == 1) & (card_gen == 0)).sum()),
          "| cardiomegaly false-positive | %d |" % (((card_true == 0) & (card_gen == 1)).sum()),
          "",
          "> ⚠️ These counts use the project's regex extractor on both sides. CheXbert",
          "> agreed with it to within 0.002 on micro-F1, so it is a fair proxy — but the",
          "> headline number to quote remains the CheXbert micro-F1-14 of **0.5939**."]

    OUT_F.write_text("\n".join(L), encoding="utf-8")
    print("  wrote %s" % OUT_F)
    print("  %d cases across %d groups" % (sum(len(g[1]) for g in groups), len(groups)))
    print()
    print("  mean findings matched: %.2f / 8" % agree.mean())
    print("  perfect matches      : %d (%.1f%%)" % ((agree == 8).sum(), (agree == 8).mean()*100))
    print("  cardiomegaly missed  : %d" % ((card_true == 1) & (card_gen == 0)).sum())


if __name__ == "__main__":
    main()
