"""
Selective deferral — the Stage 13 policy in the live system.

WHAT IT DOES
------------
Declines to answer when the prediction sits too close to the operating point,
and refers the case to a radiologist instead. "Too close" is projection-specific:

    AP (bedside)   margin cutoff 0.2247   ->  77% of films answered
    PA (standing)  margin cutoff 0.0029   ->  99.7% answered

That asymmetry is the whole point, and it was fitted, not chosen. AP films carry
less usable information (Stage 9: AUROC 0.8224 vs 0.8864), so the system must be
correspondingly more reluctant to commit on them. Deferring the same fraction of
both leaves the AP/PA accuracy gap completely intact -- measured 6.68 -> 6.28.
Deferring per projection closes it: 6.68 -> 0.78 at 85.8% coverage.

WHAT IT DOES NOT DO
-------------------
Deferral does not improve the model. It changes which cases the model answers.
Accuracy on the answered subset is NOT the system's accuracy, and every number
this module reports carries its coverage for exactly that reason.

The cutoffs were fitted on validation (n=4,474) and frozen before test was
touched. If deferral_policy.json is missing the service runs normally with
deferral disabled -- a missing analysis file must never take the demo down.
"""
from __future__ import annotations

import json
from pathlib import Path


class DeferralPolicy:
    """Decides whether the system should answer, or refer to a radiologist."""

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
        """Returns the deferral decision for one prediction.

        An unspecified projection is never deferred. Guessing a cutoff would
        either defer PA films that did not need it, or -- worse -- answer AP
        films that did.
        """
        if not self.enabled:
            return dict(active=False, defer=False)

        v = (view or "").strip().upper()
        if v not in self.cutoff:
            return dict(active=True, defer=False, view=None,
                        reason="projection not specified; answering with the "
                               "global operating point")

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
