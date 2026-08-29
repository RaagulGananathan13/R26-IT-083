"""
Component 04 — Constrained Cost-Sensitive Decision Layer (CDL).

The previous component reached its per-class targets with a hand-tuned
"STEMI-Boost": a hard-coded multiplier applied to the STEMI probability when
troponin and ST-elevation crossed chosen values.  That is a hidden classifier
with no stated objective, tuned by hand until the numbers looked acceptable.

CDL replaces it with a stated optimisation problem:

    maximise    macro-F1(argmax_k  w_k * p_k)
    subject to  recall_k >= floor   for every class k
                w in R^K_+,  w_1 = 1  (scale is unidentifiable)

Solved by multi-start random search followed by coordinate refinement on a
log-scale grid.  Two properties make it trustworthy:

  * It is fitted on VALIDATION only and then frozen.  The test fold is never
    consulted while choosing w.
  * The search is repeated over B bootstrap resamples of the validation fold
    and the component-wise MEDIAN weight vector is kept.  A single fit to ~760
    validation cases would overfit the decision boundary; the bootstrap median
    is markedly more stable and costs nothing at inference.

If the floor is infeasible the layer relaxes it through a declared ladder
(0.75 -> 0.74 -> ...) and records which rung was used, so a shortfall is
reported rather than hidden.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


# --------------------------------------------------------------------------
def _apply(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.argmax(P * w[None, :], axis=1)


def _stats(y: np.ndarray, pred: np.ndarray, K: int) -> Tuple[float, np.ndarray]:
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), np.maximum(support, 1), dtype=float)
    macro = f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)
    return float(macro), recall


def _score(y, P, w, K, floor, penalty: float = 1.0):
    """
    Feasibility-aware score: climb towards the floor, then maximise macro-F1.

    The penalty weight matters.  Set it too high (an early version used 10)
    and the search will happily destroy macro-F1 to buy a fraction of a point
    of recall on the rarest class — it once traded NSTEMI recall from 82% down
    to 70% to lift STEMI recall, for a net macro-F1 loss of 0.04.  A penalty of
    ~1 keeps the constraint meaningful without letting it dominate, and the
    caller still verifies feasibility explicitly afterwards.
    """
    macro, recall = _stats(y, _apply(P, w), K)
    gap = float(np.minimum(recall - floor, 0).sum())     # <= 0
    return (macro + penalty * gap), macro, recall


# --------------------------------------------------------------------------
def _search(y: np.ndarray, P: np.ndarray, K: int, floor: float,
            rng: np.random.RandomState, n_random: int = 4000,
            n_refine: int = 3) -> Tuple[np.ndarray, float]:
    best_w = np.ones(K)
    best_s, _, _ = _score(y, P, best_w, K, floor)

    # --- multi-start random search in log space --------------------------
    W = np.exp(rng.uniform(-2.5, 2.5, size=(n_random, K)))
    W[:, 0] = 1.0                                        # fix the scale
    for w in W:
        s, _, _ = _score(y, P, w, K, floor)
        if s > best_s:
            best_s, best_w = s, w.copy()

    # --- coordinate refinement -------------------------------------------
    for span in (0.8, 0.3, 0.1)[:n_refine]:
        improved = True
        while improved:
            improved = False
            for k in range(1, K):
                base = np.log(best_w[k])
                for delta in np.linspace(-span, span, 21):
                    w = best_w.copy()
                    w[k] = float(np.exp(base + delta))
                    s, _, _ = _score(y, P, w, K, floor)
                    if s > best_s + 1e-9:
                        best_s, best_w, improved = s, w, True
    return best_w, best_s


# --------------------------------------------------------------------------
class ConstrainedDecisionLayer:
    def __init__(self, class_names: Sequence[str], floor_ladder: Sequence[float],
                 n_bootstrap: int = 40, seed: int = 42, n_random: int = 4000,
                 verbose: bool = True, margin: float = 0.0):
        self.verbose = verbose
        self.class_names = list(class_names)
        self.K = len(class_names)
        self.floor_ladder = list(floor_ladder)
        # Search target sits above the reported floor so the constraint still
        # holds once the layer meets patients it was not fitted on.
        self.margin = float(margin)
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.n_random = n_random
        self.w: np.ndarray = np.ones(self.K)
        self.info: Dict = {}

    # ----------------------------------------------------------------
    def fit(self, P_val: np.ndarray, y_val: np.ndarray,
            groups: np.ndarray | None = None) -> "ConstrainedDecisionLayer":
        rng = np.random.RandomState(self.seed)
        P_val = np.asarray(P_val, dtype=float)
        y_val = np.asarray(y_val, dtype=int)

        chosen_floor = None      # the floor we report against
        search_floor = None      # the (higher) floor we actually optimise for
        point_w = None
        for floor in self.floor_ladder:
            target = min(0.99, floor + self.margin)
            w, _ = _search(y_val, P_val, self.K, target, rng, self.n_random)
            _, _, rec = _score(y_val, P_val, w, self.K, target)
            # Accept as soon as the REPORTED floor is cleared; clearing the
            # tightened target as well is a bonus, not a requirement.
            if rec.min() >= floor - 1e-9:
                chosen_floor, search_floor, point_w = floor, target, w
                break
        if point_w is None:                       # nothing feasible: best effort
            chosen_floor = self.floor_ladder[-1]
            search_floor = min(0.99, chosen_floor + self.margin)
            point_w, _ = _search(y_val, P_val, self.K, search_floor, rng, self.n_random)

        # --- bootstrap stabilisation --------------------------------------
        Ws: List[np.ndarray] = [point_w]
        n = len(y_val)
        if groups is not None:
            uniq = np.unique(groups)
            index_of = {g: np.where(groups == g)[0] for g in uniq}
        try:
            from progress import StepProgress
            bar = StepProgress(self.n_bootstrap, "CDL boot", enabled=self.verbose)
        except Exception:
            bar = None
        for _ in range(self.n_bootstrap):
            if bar:
                bar.step(f"floor={chosen_floor:.2f}")
            if groups is not None:
                picked = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([index_of[g] for g in picked])
            else:
                idx = rng.randint(0, n, n)
            if len(np.unique(y_val[idx])) < self.K:
                continue
            wb, _ = _search(y_val[idx], P_val[idx], self.K, search_floor, rng,
                            n_random=max(600, self.n_random // 6), n_refine=2)
            Ws.append(wb)

        w_med = np.median(np.vstack(Ws), axis=0)
        w_med = w_med / w_med[0]

        # Three candidates, judged on the full validation fold:
        #   * the unweighted argmax baseline — reweighting must EARN its place
        #   * the point estimate
        #   * the bootstrap median
        # Selection is lexicographic: prefer a candidate that meets the floor;
        # among equals, take the higher macro-F1.  Without the baseline in this
        # set the layer can be strictly worse than doing nothing and still be
        # deployed, which is exactly what happened before this was added.
        cands = {
            "argmax-baseline": np.ones(self.K),
            "point-estimate": point_w,
            "bootstrap-median": w_med,
        }
        scored = {}
        for name, w in cands.items():
            _, macro, rec = _score(y_val, P_val, w, self.K, chosen_floor)
            # prefer candidates that also clear the tightened target
            scored[name] = (bool(rec.min() >= chosen_floor - 1e-9), float(macro),
                            rec, w, bool(rec.min() >= search_floor - 1e-9))
        # lexicographic: clears reported floor > clears tightened target > macro-F1
        variant = max(scored, key=lambda k: (scored[k][0], scored[k][4], scored[k][1]))
        feasible, macro, rec, self.w, _tight = scored[variant]

        self.info = {
            "weights": {c: float(v) for c, v in zip(self.class_names, self.w)},
            "floor_requested": float(self.floor_ladder[0]),
            "floor_achieved": float(chosen_floor),
            "floor_search_target": float(search_floor),
            "margin": float(self.margin),
            "floor_relaxed": bool(chosen_floor != self.floor_ladder[0]),
            "floor_met_on_val": bool(feasible),
            "variant": variant,
            "reweighting_used": bool(variant != "argmax-baseline"),
            "candidates_val": {k: {"meets_floor": v[0], "macro_f1": v[1]}
                               for k, v in scored.items()},
            "n_bootstrap": len(Ws) - 1,
            "val_macro_f1": float(macro),
            "val_recall": {c: float(v) for c, v in zip(self.class_names, rec)},
            "val_min_recall": float(rec.min()),
            "weight_dispersion_iqr": {
                c: float(np.subtract(*np.percentile(np.vstack(Ws)[:, i], [75, 25])))
                for i, c in enumerate(self.class_names)
            },
        }
        return self

    # ----------------------------------------------------------------
    def predict(self, P: np.ndarray) -> np.ndarray:
        return _apply(np.asarray(P, dtype=float), self.w)

    def transform_proba(self, P: np.ndarray) -> np.ndarray:
        """Re-weighted, renormalised probabilities (for display / SHAP)."""
        Q = np.asarray(P, dtype=float) * self.w[None, :]
        return Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)

    def report(self) -> str:
        i = self.info
        lines = [
            f"  selected variant           {i['variant']}  "
            f"({i['n_bootstrap']} bootstrap fits)",
            "  candidates on validation:",
        ]
        for k, v in i["candidates_val"].items():
            mark = " <- selected" if k == i["variant"] else ""
            lines.append(f"      {k:<18} macro-F1={v['macro_f1']:.4f}  "
                         f"floor={'met' if v['meets_floor'] else 'missed'}{mark}")
        lines += [
            f"  recall floor requested     {i['floor_requested']:.2f}",
            f"  recall floor searched at   {i['floor_search_target']:.2f}"
            f"   (= {i['floor_achieved']:.2f} + {i['margin']:.2f} margin)"
            + ("   [LADDER RELAXED]" if i["floor_relaxed"] else ""),
            f"  floor met on validation    {i['floor_met_on_val']}",
            f"  validation macro-F1        {i['val_macro_f1']:.4f}",
            "  class weights (w):",
        ]
        for c in self.class_names:
            lines.append(f"      {c:<10} w={i['weights'][c]:8.4f}   "
                         f"val recall={i['val_recall'][c]*100:6.2f}%   "
                         f"IQR={i['weight_dispersion_iqr'][c]:.3f}")
        return "\n".join(lines)
