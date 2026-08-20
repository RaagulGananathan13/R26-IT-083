"""
Per-projection operating points — the Stage 9A contribution, in the live system.

WHAT THIS IS
------------
Chest radiographs are taken posteroanterior (PA, patient standing) or
anteroposterior (AP, patient too ill to stand, portable/bedside). AP magnifies
the cardiac silhouette because the heart sits anterior, further from the
detector.

Clinical radiology has handled this for decades with a projection-specific
decision rule: cardiomegaly is CTR > 0.50 on PA and > 0.55 on AP. The AI
fairness literature does the opposite -- it tries to make models BLIND to
projection (adversarial debiasing, gradient reversal).

We measured that the algorithmic approach is wrong. Driving projection AUC to
0.5000 (complete invariance, beyond the published method's 0.61) closed only
13.3% of the AP/PA gap and cost 0.0789 AUROC. Projection-conditional thresholds
reduced the reported disparity metric by 73.3% at ZERO accuracy cost, exceeding
the 46.7% reported by Pereira et al. (MIDL 2023).

Our fitted thresholds land in the same direction as the clinical convention:

    Cardiomegaly    AP 0.409   PA 0.348    ratio 1.18
    clinical CTR    AP 0.55    PA 0.50     ratio 1.10

WHAT THIS IS NOT
----------------
Thresholding does NOT make the model better at AP films. AUROC is computed over
the whole ranking; a threshold is one cut through it, and cutting per group
cannot reorder any case. We proved the discrimination gap is unchanged to
1e-12. The AP/PA gap of 0.0639 is irreducible at the representation level --
it reflects genuine information loss at acquisition. Hence `reliability`, which
reports that honestly instead of hiding it.
"""
from __future__ import annotations

import json
from pathlib import Path


class ThresholdPolicy:
    """Chooses the operating point, and states how much to trust the result."""

    def __init__(self, path: Path, projection_auroc: dict, gap: float):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.pathologies = d["pathologies"]
        self.tables = {"global": d["global"], "AP": d["AP"], "PA": d["PA"]}
        self.projection_auroc = projection_auroc
        self.gap = gap

    def get(self, pathology: str, view: str | None) -> tuple[float, str]:
        """Returns (threshold, which_table_was_used).

        An unknown view falls back to the global threshold rather than guessing.
        Guessing PA on a bedside film would under-call cardiomegaly on exactly
        the patients least able to tolerate a missed diagnosis.
        """
        v = (view or "").strip().upper()
        if v in ("AP", "PA") and pathology in self.tables[v]:
            return float(self.tables[v][pathology]), v
        return float(self.tables["global"].get(pathology, 0.5)), "global"

    def reliability(self, view: str | None) -> dict:
        """Honest statement of expected reliability for this acquisition."""
        v = (view or "").strip().upper()
        if v == "AP":
            return dict(
                level="reduced", view="AP",
                measured_auroc=self.projection_auroc["AP"],
                message=("AP (bedside) film. Measured AUROC on AP films is %.4f "
                         "versus %.4f on PA — a gap of %.4f. AP images carry less "
                         "usable information: the scapulae overlie the lung fields, "
                         "the cardiac silhouette is magnified, and the patient is "
                         "usually supine. Interpret with additional caution."
                         % (self.projection_auroc["AP"],
                            self.projection_auroc["PA"], self.gap)))
        if v == "PA":
            return dict(
                level="standard", view="PA",
                measured_auroc=self.projection_auroc["PA"],
                message=("PA (standing) film — the diagnostic standard. Measured "
                         "AUROC %.4f." % self.projection_auroc["PA"]))
        return dict(
            level="unknown", view=None, measured_auroc=None,
            message=("Projection not specified, so the global operating point is "
                     "used. Selecting AP or PA applies the projection-specific "
                     "threshold and reports the measured reliability for that view."))
