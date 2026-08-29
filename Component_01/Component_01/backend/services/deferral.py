"""
Decides when the model should refuse to answer and ask for a radiologist.

If a prediction sits very close to the decision threshold, the model is
basically guessing. Rather than commit, we flag it for review.

How close counts as "too close" depends on the view:

    AP (bedside)   cutoff 0.2247   ->  we answer about 77% of these
    PA (standing)  cutoff 0.0029   ->  we answer about 99.7% of these

That difference is the whole point, and it came out of the validation data
rather than being picked by hand. AP films carry less usable information (AUROC
0.8224 vs 0.8864), so the model should be more reluctant to commit on them.

We tested the obvious alternative first. Deferring the same share of both types
barely helps the AP/PA accuracy gap at all (6.68 -> 6.28 points). Deferring more
on AP closes it (6.68 -> 0.78).

Worth remembering: this does not make the model more accurate. It changes which
cases the model is willing to answer. So any accuracy number from this has to be
quoted together with its coverage.

If the policy file is missing we just run with deferral switched off. A missing
analysis file should never take the whole demo down.
"""
from __future__ import annotations

import json
from pathlib import Path


class DeferralPolicy:
    """Decides whether to answer, or hand the case to a radiologist."""

    def __init__(self, path: Path):
        self.enabled = False
        self.cutoff: dict[str, float] = {}
        self.stats: dict = {}
        self.coverage = None

        p = Path(path)
        if not p.exists():
            print("[deferral] %s not found -- deferral disabled. "
                  "Run: python stage13_deferral.py" % p.name)
            return

        d = json.loads(p.read_text(encoding="utf-8"))
        self.cutoff = {k: float(v) for k, v in d["margin_cutoff"].items()}
        self.stats = d.get("measured_on_test", {})
        self.coverage = d.get("deploy_coverage")
        self.enabled = True
        print("[deferral] policy loaded -- AP cutoff %.4f / PA cutoff %.4f "
              "(target coverage %.0f%%)"
              % (self.cutoff.get("AP", 0), self.cutoff.get("PA", 0),
                 100 * (self.coverage or 0)))

    def assess(self, prob: float, threshold: float, view: str | None) -> dict:
        """Work out whether this one prediction is too close to call.

        If the view wasn't given we never defer. Picking a cutoff by guessing
        would either flag PA films that were fine, or worse, wave through AP
        films that should have been checked.
        """
        if not self.enabled:
            return dict(active=False, defer=False)

        v = (view or "").strip().upper()
        if v not in self.cutoff:
            return dict(active=True, defer=False, view=None,
                        reason="projection not specified; answering with the "
                               "global operating point")

        # How far the score sits from the line. Small margin = coin flip.
        margin = abs(float(prob) - float(threshold))
        cut = self.cutoff[v]
        defer = margin < cut

        out = dict(active=True, defer=bool(defer), view=v,
                   margin=round(margin, 4), cutoff=round(cut, 4))
        if defer:
            out["reason"] = (
                "The prediction (%.3f) sits within %.3f of this view's decision "
                "threshold (%.3f) — too close to call. On %s films the system "
                "refers cases this uncertain rather than committing."
                % (prob, cut, threshold, v))
        if self.stats:
            out["measured"] = dict(
                coverage=round(self.stats.get("coverage", 0), 1),
                accuracy=round(self.stats.get("accuracy", 0), 2),
                sensitivity=round(self.stats.get("sensitivity", 0), 1),
                coverage_ap=round(self.stats.get("coverage_ap", 0), 1),
                coverage_pa=round(self.stats.get("coverage_pa", 0), 1),
                gap=round(self.stats.get("gap", 0), 2))
        return out
