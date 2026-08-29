"""
Picks the decision threshold based on how the X-ray was taken.

Background: chest X-rays come in two types. PA means the patient stood up in the
radiology department. AP means a portable machine came to their bed, which
happens when they are too sick to walk. On an AP film the heart sits further
from the detector, so it looks bigger than it really is.

Because of that we use a different cut-off for each type:

    Cardiomegaly    AP 0.409   PA 0.348

Radiologists have done the same thing for decades. The textbook rule is a
cardiothoracic ratio above 0.50 on PA but above 0.55 on AP. Our numbers were
fitted from validation data and came out pointing the same way.

One thing to be clear about: changing the threshold does NOT make the model
better on AP films. It only changes where we draw the line. The model still
scores 0.8224 on AP and 0.8864 on PA, and that gap does not move. That is why
this file also reports a reliability level, so the UI can be honest about it.
"""
from __future__ import annotations

import json
from pathlib import Path


class ThresholdPolicy:
    """Chooses the cut-off, and says how much to trust the result."""

    def __init__(self, path: Path, projection_auroc: dict, gap: float):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.pathologies = d["pathologies"]
        self.tables = {"global": d["global"], "AP": d["AP"], "PA": d["PA"]}
        self.projection_auroc = projection_auroc
        self.gap = gap

    def get(self, pathology: str, view: str | None) -> tuple[float, str]:
        """Returns (threshold, which table it came from).

        If we don't know the view we fall back to the global threshold rather
        than guessing. Guessing PA on a bedside film would raise the bar on
        exactly the patients we least want to miss.
        """
        v = (view or "").strip().upper()
        if v in ("AP", "PA") and pathology in self.tables[v]:
            return float(self.tables[v][pathology]), v
        return float(self.tables["global"].get(pathology, 0.5)), "global"

    def reliability(self, view: str | None) -> dict:
        """How much the user should trust this result, given the view."""
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
