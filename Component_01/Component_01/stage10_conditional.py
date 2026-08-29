"""
COMPONENT_01 · STAGE 10 · ACQUISITION-CONDITIONAL SPECIALISATION
================================================================

THE ARGUMENT
------------
Stage 9B established that projection INVARIANCE fails on this axis: driving
projection AUC to 0.5000 (complete invariance, beyond the published method's
0.61) closed only 13.3% of the AP/PA gap and cost 0.0789 AUROC. The gap is not
a learned shortcut -- AP films are intrinsically harder to read.

If the two projections are genuinely different problems, the remedy is not to
hide the difference but to let the model SPECIALISE on each. One shared head
must compromise between two distributions whose statistics differ sharply
(cardiomegaly prevalence 62% AP vs 32% PA), and that compromise costs accuracy
on BOTH groups.

    invariance  -> remove acquisition -> levelling down (measured: AUROC 0.4650)
    conditional -> exploit acquisition -> positive sum  (this file's hypothesis)

CRITICAL DESIGN CHOICE: IDENTITY INITIALISATION
-----------------------------------------------
Every conditional variant is constructed to be EXACTLY EQUIVALENT to the
Stage 5 baseline at initialisation:

  * per_projection : both heads are initialised from the SAME Stage 5 head,
                     so before any training the model reproduces best.pt bit
                     for bit, then diverges only where specialisation helps.
  * film           : the final FiLM layer is zero-initialised, so gamma=0 and
                     beta=0 give feat*(1+0)+0 = feat -- again exactly best.pt.

This matters for more than tidiness. A randomly-initialised head would start
far WORSE than the baseline, and several epochs would be spent merely climbing
back. Any measured gain could then be recovery rather than specialisation, and
a null result would be uninterpretable. Identity init means the run starts at
the baseline and every subsequent point is attributable to conditioning.

WHAT 10A TESTS BEFORE ANY TRAINING IS PAID FOR
----------------------------------------------
Frozen Stage 5 features, three linear probes:

    A  shared        1024-d features -> one head        (the baseline)
    B  shared+acq    features ++ acquisition vector     (cheapest conditioning)
    C  per-projection separate AP and PA heads          (full specialisation)

If C cannot beat A on frozen features, specialisation does not help and the
expensive fine-tuning arms are cancelled. Note C is handicapped on purpose: it
splits the training data between two heads, so each sees roughly half. Winning
DESPITE that is strong evidence; losing narrowly is ambiguous and is reported
as such rather than spun.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PATHOLOGIES = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
               "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]
ACQ_FEATURES = ["is_AP", "is_portable", "off_hours", "insp_lungfrac",
                "pen_mean", "pen_contrast", "rot_lr_asym", "blur_lapvar"]


# ====================================================================
# 1 · model
# ====================================================================
def _stage5_head(d: int, n: int, p_drop: float = 0.3) -> nn.Sequential:
    """Byte-compatible replica of the Stage 5 classifier head."""
    return nn.Sequential(
        nn.LayerNorm(d), nn.Dropout(p_drop), nn.Linear(d, 512),
        nn.GELU(), nn.Dropout(p_drop * 0.66), nn.Linear(512, n))


class CXRConditional(nn.Module):
    """Stage 5 trunk + an acquisition-conditional head.

    mode = 'shared'          -> identical to Stage 5 (control arm)
           'per_projection'  -> separate AP / PA heads, both init from Stage 5
           'film'            -> acquisition vector modulates features, zero-init

    The trunk (`features`, `avgpool`) and the `classifier` module are named
    exactly as in Stage 5 so best.pt loads by key. DO NOT rebuild this from
    torchvision's ConvNeXt with a swapped head: torchvision applies the
    classifier to the 4-D (B,1024,1,1) avgpool output while Stage 5 flattens
    FIRST, and nn.LayerNorm(1024) on a 4-D tensor silently normalises a size-1
    dimension instead of the channels.
    """

    def __init__(self, n_path: int = 8, mode: str = "per_projection",
                 n_acq: int = len(ACQ_FEATURES), p_drop: float = 0.3):
        super().__init__()
        if mode not in {"shared", "per_projection", "film"}:
            raise ValueError("unknown mode " + repr(mode))
        import torchvision
        base = torchvision.models.convnext_base(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        d = base.classifier[2].in_features                 # 1024
        self.feat_dim, self.n_path, self.mode = d, n_path, mode

        self.classifier = _stage5_head(d, n_path, p_drop)  # loads from best.pt

        if mode == "per_projection":
            # Second head for AP films. Weights are copied from `classifier`
            # in load_stage5, so both start identical to the baseline.
            self.classifier_ap = _stage5_head(d, n_path, p_drop)
        elif mode == "film":
            self.film = nn.Sequential(
                nn.Linear(n_acq, 128), nn.GELU(), nn.Linear(128, 2 * d))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)             # -> identity at init

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor, acq: torch.Tensor | None = None,
                is_ap: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x)

        if self.mode == "shared":
            return self.classifier(feat)

        if self.mode == "film":
            if acq is None:
                raise ValueError("film mode requires the acquisition vector")
            gamma, beta = self.film(acq.to(feat.dtype)).chunk(2, dim=-1)
            # FiLM is applied AFTER the head's LayerNorm, never before.
            # LayerNorm subtracts the mean and divides by the std, so a
            # modulation applied beforehand is exactly cancelled:
            #   ((1+g)x + b - (1+g)m - b) / ((1+g)s) = (x - m)/s
            # Placing FiLM first silently neuters the entire mechanism -- the
            # model trains, the loss falls, and the conditioning does nothing.
            h = self.classifier[0](feat)                    # LayerNorm
            h = h * (1.0 + gamma) + beta                    # modulate
            for layer in self.classifier[1:]:               # Dropout->Linear->...
                h = layer(h)
            return h

        # per_projection: route each sample to its own head, then recombine.
        # Both heads run on the full batch. Masked gather would be marginally
        # cheaper but produces empty sub-batches when a batch happens to be
        # all-AP or all-PA, and LayerNorm on an empty batch raises.
        if is_ap is None:
            raise ValueError("per_projection mode requires is_ap")
        m = is_ap.reshape(-1, 1).to(feat.dtype)
        return m * self.classifier_ap(feat) + (1.0 - m) * self.classifier(feat)

    # ---------------- checkpoint plumbing ----------------
    def load_stage5(self, ck: dict, use_ema: bool = True) -> dict:
        """Load Stage 5, then make every conditional path an identity copy."""
        sd = {(k[7:] if k.startswith("module.") else k): v
              for k, v in ck["model"].items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has keys this model lacks: {unexpected[:8]}")
        allowed = {n for n, _ in self.named_parameters()
                   if n.startswith(("classifier_ap", "film"))}
        allowed |= {n for n, _ in self.named_buffers()
                    if n.startswith(("classifier_ap", "film"))}
        stray = set(missing) - allowed
        if stray:
            raise RuntimeError(f"trunk/head weights missing from checkpoint: {sorted(stray)[:8]}")
        if use_ema and ck.get("ema"):
            msd = self.state_dict()
            bad = [k for k in ck["ema"] if k not in msd]
            if bad:
                raise RuntimeError(f"EMA keys absent from model: {bad[:5]}")
            for k, v in ck["ema"].items():
                msd[k].copy_(v)
        if self.mode == "per_projection":
            # AFTER EMA, so the AP head copies the weights actually used.
            self.classifier_ap.load_state_dict(self.classifier.state_dict())
        return dict(loaded=len(sd), fresh=len(missing), mode=self.mode)

    def param_groups(self, lr_trunk: float, lr_head: float):
        trunk, head = [], []
        for n, p in self.named_parameters():
            (head if n.startswith(("classifier", "film")) else trunk).append(p)
        return [{"params": trunk, "lr": lr_trunk}, {"params": head, "lr": lr_head}]


# ====================================================================
# 2 · the 10A gate — linear probes on frozen features
# ====================================================================
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _fit_predict(Xtr, ytr, Xte, seed=0, C=1.0, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float("nan"))
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    p.fit(Xtr, ytr)
    return p.predict_proba(Xte)[:, 1]


def probe_shared(Xtr, Ytr, Xte, Yte, pathologies=PATHOLOGIES, **kw) -> dict:
    out = {k: _auroc(Yte[k], _fit_predict(Xtr, Ytr[k].to_numpy(), Xte, **kw))
           for k in pathologies}
    return out


def probe_conditional(Xtr, Ytr, aptr, Xte, Yte, apte,
                      pathologies=PATHOLOGIES, **kw) -> dict:
    """Separate AP / PA heads.

    Deliberately handicapped: each head trains on roughly half the data.
    Winning despite that is strong evidence for specialisation.
    """
    aptr, apte = np.asarray(aptr).astype(bool), np.asarray(apte).astype(bool)
    out = {}
    for k in pathologies:
        s = np.empty(len(Xte), dtype=np.float64)
        for m_tr, m_te in ((aptr, apte), (~aptr, ~apte)):
            if m_te.sum() == 0:
                continue
            s[m_te] = _fit_predict(Xtr[m_tr], Ytr[k].to_numpy()[m_tr], Xte[m_te], **kw)
        out[k] = _auroc(Yte[k], s)
    return out


def compare_probes(Xtr, Ytr, Atr, Xte, Yte, Ate,
                   pathologies=PATHOLOGIES, acq_features=ACQ_FEATURES, **kw) -> dict:
    """Three-arm probe ablation. Arm C beating arm A is the GO signal."""
    aptr = (Atr["is_AP"] > 0.5).to_numpy()
    apte = (Ate["is_AP"] > 0.5).to_numpy()
    Xtr_a = np.hstack([Xtr, np.nan_to_num(Atr[acq_features].to_numpy(np.float64))])
    Xte_a = np.hstack([Xte, np.nan_to_num(Ate[acq_features].to_numpy(np.float64))])

    res = {"A_shared": probe_shared(Xtr, Ytr, Xte, Yte, pathologies, **kw),
           "B_shared_plus_acq": probe_shared(Xtr_a, Ytr, Xte_a, Yte, pathologies, **kw),
           "C_per_projection": probe_conditional(Xtr, Ytr, aptr, Xte, Yte, apte,
                                                 pathologies, **kw)}
    out = {}
    for tag, per in res.items():
        mean = float(np.nanmean([per[k] for k in pathologies]))
        out[tag] = dict(per_pathology=per, mean_auroc=mean)
    a, c = out["A_shared"]["mean_auroc"], out["C_per_projection"]["mean_auroc"]
    b = out["B_shared_plus_acq"]["mean_auroc"]
    out["verdict"] = dict(
        gain_conditional=float(c - a), gain_acq_features=float(b - a),
        best=max(("A_shared", a), ("B_shared_plus_acq", b),
                 ("C_per_projection", c), key=lambda t: t[1])[0],
        go=bool(max(b, c) - a > 0.002))
    return out


# ====================================================================
# 3 · self-tests
# ====================================================================
def _selftest(verbose: bool = True) -> tuple[int, int]:
    P, F = [], []

    def g(name, ok, extra=""):
        (P if ok else F).append(name)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<60}{extra}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    xb = torch.randn(4, 3, 64, 64)
    ap = torch.tensor([1.0, 0.0, 1.0, 0.0])
    acq = torch.randn(4, len(ACQ_FEATURES))

    ref = CXRConditional(8, "shared").eval()
    sd_disease = {k: v for k, v in ref.state_dict().items()
                  if not k.startswith(("classifier_ap", "film"))}

    # ---- identity initialisation: the property everything depends on ----
    for mode in ("per_projection", "film"):
        m = CXRConditional(8, mode)
        info = m.load_stage5({"model": sd_disease}, use_ema=False)
        m.eval()
        with torch.no_grad():
            base = ref(xb)
            got = m(xb, acq=acq, is_ap=ap)
        g(f"{mode}: loads Stage 5 (fresh keys only in the new module)",
          info["loaded"] > 300 and info["fresh"] > 0, f"{info}")
        g(f"*** {mode}: EXACTLY reproduces the baseline at init ***",
          torch.allclose(base, got, atol=1e-6),
          f"max|diff|={float((base-got).abs().max()):.2e}")

    # ---- per-projection routing ----
    m = CXRConditional(8, "per_projection")
    m.load_stage5({"model": sd_disease}, use_ema=False)
    with torch.no_grad():
        m.classifier_ap[5].bias.add_(10.0)          # make the AP head distinct
    m.eval()
    with torch.no_grad():
        out = m(xb, is_ap=ap)
        base = ref(xb)
    g("per_projection: AP rows use the AP head",
      bool(((out[0] - base[0]).abs().mean() > 5).item()))
    g("per_projection: PA rows are untouched",
      torch.allclose(out[1], base[1], atol=1e-6))
    with torch.no_grad():
        all_pa = m(xb, is_ap=torch.zeros(4))
    g("per_projection: an all-PA batch never touches the AP head",
      torch.allclose(all_pa, base, atol=1e-6))
    with torch.no_grad():
        all_ap = m(xb, is_ap=torch.ones(4))
    g("per_projection: an all-AP batch does not crash (no empty sub-batch)",
      tuple(all_ap.shape) == (4, 8))

    # ---- film ----
    mf = CXRConditional(8, "film")
    mf.load_stage5({"model": sd_disease}, use_ema=False)
    with torch.no_grad():
        g("film: final layer is zero-initialised",
          float(mf.film[-1].weight.abs().sum()) == 0.0
          and float(mf.film[-1].bias.abs().sum()) == 0.0)
    # A UNIFORM modulation must survive, which it only does because FiLM is
    # applied after the LayerNorm. Placed before, LayerNorm cancels it exactly
    # and this test fails -- which is how the bug was found.
    with torch.no_grad():
        mf.film[-1].bias.add_(0.5)
    mf.eval()
    with torch.no_grad():
        g("film: a UNIFORM modulation survives (not cancelled by LayerNorm)",
          not torch.allclose(mf(xb, acq=acq), ref(xb), atol=1e-4),
          f"max|diff|={float((mf(xb, acq=acq)-ref(xb)).abs().max()):.3f}")
    # And a PER-CHANNEL modulation must also work.
    mf2 = CXRConditional(8, "film")
    mf2.load_stage5({"model": sd_disease}, use_ema=False)
    with torch.no_grad():
        mf2.film[-1].bias[::2].add_(0.7)
    mf2.eval()
    with torch.no_grad():
        g("film: a PER-CHANNEL modulation changes the output",
          not torch.allclose(mf2(xb, acq=acq), ref(xb), atol=1e-4))

    # ---- guards ----
    for mode, kw, msg in (("per_projection", dict(acq=acq), "is_ap"),
                          ("film", dict(is_ap=ap), "acquisition")):
        mm = CXRConditional(8, mode).eval()
        try:
            mm(xb, **kw); ok = False
        except ValueError:
            ok = True
        g(f"{mode}: raises when its required input is missing", ok)
    try:
        CXRConditional(8, "nonsense"); ok = False
    except ValueError:
        ok = True
    g("rejects an unknown mode", ok)
    try:
        CXRConditional(8, "film").load_stage5({"model": {"bogus.key": torch.zeros(1)}})
        ok = False
    except RuntimeError:
        ok = True
    g("rejects an unexpected checkpoint key (fails loud)", ok)

    g("param_groups splits trunk from head",
      len(CXRConditional(8, "film").param_groups(1e-5, 1e-4)) == 2)

    # ---- probes: separable-by-group synthetic data ----
    n, d = 3000, 24
    apn = rng.random(n) < 0.6
    y = (rng.random(n) < 0.4).astype(int)
    X = rng.normal(0, 1, (n, d))
    # The SAME label is encoded on different axes for AP and PA, so one shared
    # linear head cannot fit both while two specialised heads can.
    X[apn, 0] += 2.0 * y[apn]
    X[~apn, 1] += 2.0 * y[~apn]
    half = n // 2
    Ytr = pd.DataFrame({"D": y[:half]}); Yte = pd.DataFrame({"D": y[half:]})
    Atr = pd.DataFrame({f: np.zeros(half) for f in ACQ_FEATURES})
    Ate = pd.DataFrame({f: np.zeros(n - half) for f in ACQ_FEATURES})
    Atr["is_AP"] = apn[:half].astype(float)
    Ate["is_AP"] = apn[half:].astype(float)
    R = compare_probes(X[:half], Ytr, Atr, X[half:], Yte, Ate, ["D"])
    g("probes produce all three arms plus a verdict",
      set(R) == {"A_shared", "B_shared_plus_acq", "C_per_projection", "verdict"})
    g("*** conditional probe beats shared on group-specific signal ***",
      R["verdict"]["gain_conditional"] > 0.02,
      f"A={R['A_shared']['mean_auroc']:.4f} -> C={R['C_per_projection']['mean_auroc']:.4f}")
    g("verdict flags GO on a real gain", R["verdict"]["go"])

    # ---- probes: null case, identical rule in both groups -> no gain ----
    X0 = rng.normal(0, 1, (n, d)); X0[:, 0] += 2.0 * y
    R0 = compare_probes(X0[:half], Ytr, Atr, X0[half:], Yte, Ate, ["D"])
    g("*** conditional probe does NOT invent a gain when none exists ***",
      R0["verdict"]["gain_conditional"] < 0.02,
      f"gain={R0['verdict']['gain_conditional']:+.4f}")

    if verbose:
        print(f"\n  {len(P)} passed, {len(F)} failed")
        for f in F:
            print(f"    - {f}")
    return len(P), len(F)


if __name__ == "__main__":
    print("=" * 80)
    print(" STAGE 10 · stage10_conditional.py self-test")
    print("=" * 80)
    p, f = _selftest()
    raise SystemExit(1 if f else 0)
