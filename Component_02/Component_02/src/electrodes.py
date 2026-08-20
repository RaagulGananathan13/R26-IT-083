"""
electrodes.py — Detection of limb-electrode reversal.

WHY THIS EXISTS
---------------
Electrode misplacement is one of the most common errors in clinical ECG
recording, and it is INVISIBLE to signal-quality assessment: a reversed
recording is perfectly clean, perfectly formed, and completely wrong. Our own
quality gate accepts 247/250 reversed records with a mean quality index of 1.000.

The clinical consequence is documented: right-arm/left-leg reversal mimics
inferior infarction; right-arm/left-arm reversal inverts lead I and mimics
lateral changes or dextrocardia.

The statistical consequence is the reason this module sits next to the conformal
layer: the guarantee was calibrated on correctly-placed ECGs. A reversed
recording is out of that distribution, so exchangeability fails and the promised
miss-rate bound no longer holds — while the system continues to state it.

THE MATHS
---------
Limb reversals are EXACT linear maps, because the augmented leads are derived:
    I = LA-RA,  II = LL-RA,  III = LL-LA
    aVR = -(I+II)/2,  aVL = I - II/2,  aVF = II - I/2
Swapping two electrodes therefore permutes and negates known leads. Precordial
leads are unaffected: Wilson's central terminal is (RA+LA+LL)/3, which is
invariant under a limb swap.

    RA/LA :  I -> -I,   II <-> III,  aVR <-> aVL,  aVF unchanged
    RA/LL :  I -> -III, II -> -II,   III -> -I,    aVR <-> aVF,  aVL unchanged
    LA/LL :  I <-> II,  III -> -III, aVL <-> aVF,  aVR UNCHANGED

That last line is why LA/LL is the hard one: the single most informative
detection signal — aVR — is untouched by it.

DETECTION
---------
Physiology, not machine learning. In sinus rhythm the cardiac axis points down
and to the left, away from the right shoulder, so aVR is predominantly NEGATIVE
in essentially every normal heart, and lead I is predominantly positive. A
positive aVR with an inverted lead I is the classic signature of RA/LA reversal.

Lead order assumed: I II III aVR aVL aVF V1 V2 V3 V4 V5 V6.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

I, II, III, AVR, AVL, AVF = 0, 1, 2, 3, 4, 5


# ── exact reversal simulators (for evaluation and for unit tests) ─────────
def swap_ra_la(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, I] = -x[:, I]
    y[:, II], y[:, III] = x[:, III], x[:, II]
    y[:, AVR], y[:, AVL] = x[:, AVL], x[:, AVR]
    return y


def swap_ra_ll(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, I] = -x[:, III]
    y[:, II] = -x[:, II]
    y[:, III] = -x[:, I]
    y[:, AVR], y[:, AVF] = x[:, AVF], x[:, AVR]
    return y


def swap_la_ll(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, I], y[:, II] = x[:, II], x[:, I]
    y[:, III] = -x[:, III]
    y[:, AVL], y[:, AVF] = x[:, AVF], x[:, AVL]
    return y


REVERSALS = {
    "RA/LA": swap_ra_la,
    "RA/LL": swap_ra_ll,
    "LA/LL": swap_la_ll,
}


@dataclass
class ElectrodeReport:
    suspected: bool = False
    reversal: Optional[str] = None
    confidence: float = 0.0
    reasons: list = None
    metrics: Dict[str, float] = None

    def to_dict(self):
        d = asdict(self)
        d["reasons"] = self.reasons or []
        return d


def _net_deflection(lead: np.ndarray) -> float:
    """Net signed area — positive if the lead is predominantly upright.

    Uses the signed sum rather than peak amplitude so that a tall R wave with a
    deep S wave is scored by its net direction, which is what the cardiac axis
    argument is about.
    """
    return float(np.sum(lead))


def _dominant_polarity(lead: np.ndarray) -> float:
    """Robust polarity in [-1, 1]: sign of the largest absolute excursion."""
    lo, hi = float(lead.min()), float(lead.max())
    denom = abs(hi) + abs(lo)
    return 0.0 if denom < 1e-9 else (abs(hi) - abs(lo)) / denom


def detect(signal_mv: np.ndarray) -> ElectrodeReport:
    """Screen a (n_samples, 12) millivolt ECG for limb-electrode reversal.

    Deliberately conservative: it raises suspicion, it does not assert. The
    caller decides whether to refuse or warn (see pipeline.analyse).
    """
    x = np.asarray(signal_mv, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 6:
        return ElectrodeReport(False, None, 0.0, ["not a 12-lead record"], {})

    avr_pol = _dominant_polarity(x[:, AVR])
    i_pol = _dominant_polarity(x[:, I])
    avr_net = _net_deflection(x[:, AVR])
    i_net = _net_deflection(x[:, I])

    metrics = {"aVR_polarity": avr_pol, "I_polarity": i_pol,
               "aVR_net": avr_net, "I_net": i_net}
    reasons, score, kind = [], 0.0, None

    # ── RA/LA reversal ───────────────────────────────────────────────────
    # Classic signature: aVR becomes upright AND lead I inverts. In sinus
    # rhythm aVR is negative in essentially every normal heart.
    if avr_pol > 0.15 and i_pol < -0.10:
        score = min(1.0, (avr_pol + abs(i_pol)))
        kind = "RA/LA"
        reasons.append(
            f"aVR is predominantly UPRIGHT (polarity {avr_pol:+.2f}) while lead I "
            f"is INVERTED (polarity {i_pol:+.2f}). In sinus rhythm the cardiac "
            f"axis points away from the right shoulder, so this pattern is "
            f"almost always right-arm / left-arm electrode reversal — the "
            f"alternative explanations are dextrocardia or lead misconnection.")
    elif avr_pol > 0.30:
        score = min(1.0, avr_pol)
        kind = "limb (unspecified)"
        reasons.append(
            f"aVR is predominantly upright (polarity {avr_pol:+.2f}), which is "
            f"physiologically unusual and suggests a limb-electrode problem.")

    # ── RA/LL reversal ───────────────────────────────────────────────────
    # aVR and aVF exchange, so aVF turns markedly negative while aVR rises.
    avf_pol = _dominant_polarity(x[:, AVF])
    metrics["aVF_polarity"] = avf_pol
    if kind is None and avf_pol < -0.35 and avr_pol > 0.0:
        score = min(1.0, abs(avf_pol))
        kind = "RA/LL"
        reasons.append(
            f"aVF is markedly negative (polarity {avf_pol:+.2f}) with a "
            f"non-negative aVR ({avr_pol:+.2f}). Right-arm / left-leg reversal "
            f"exchanges these two leads and mimics inferior infarction.")

    return ElectrodeReport(
        suspected=kind is not None,
        reversal=kind,
        confidence=float(round(score, 3)),
        reasons=reasons,
        metrics={k: float(round(v, 4)) for k, v in metrics.items()},
    )


LIMITATION = (
    "Left-arm / left-leg reversal leaves aVR unchanged and is therefore NOT "
    "reliably detectable by polarity rules. Published neural detectors reach "
    "only ~22% sensitivity on it, against ~90% for the other reversals. This "
    "detector does not claim to find it."
)
