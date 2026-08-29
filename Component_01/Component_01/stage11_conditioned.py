"""
Report generation, conditioned on what the classifier found.

WHY WE DO THIS

The classifier is good at spotting things (mean AUROC 0.8554, cardiomegaly
0.9189). The report generator is much weaker -- its whole advantage over just
printing a fixed paragraph was only +0.0149, which means it was barely reading
the image at all.

We measured the gap:

    report generator clinical F1                  0.5799
    what it would score if it just said what
    the classifier already knows                  0.6535
    so there is headroom of                       +0.0736

So instead of asking the decoder to work out the pathology from pixels all over
again, we simply tell it what the classifier found, as a short line of text in
front of the image tokens.

NO NEW PARAMETERS

The prompt is ordinary text. We tokenise it and run it through BART's own
embedding table, then stick it in front of the visual tokens. Nothing new is
added to the model, which has three useful consequences:

  * the older checkpoint still loads with strict=True, no missing keys
  * with an empty prompt the model is bit-for-bit the same as before, so any
    change we measure comes from the prompt and not from some newly initialised
    layer settling down
  * BioBART already knows what "pleural effusion" means from its medical
    pretraining, whereas a learned label embedding would start from nothing

WE USE PREDICTIONS, NOT THE TRUE LABELS

Training uses whatever the classifier actually predicted, not the ground truth.
If we trained on ground truth the decoder would learn to trust a perfect oracle
that does not exist at inference time, and it would fall apart precisely when
the classifier is wrong. The dropout on the prompt (PROMPT_DROPOUT) also stops
the decoder from just copying the prompt and ignoring the image.

WHAT MUST NOT BREAK

Prior-study hallucination is at 0.0000 and has to stay there. The training
targets are still the cleaned corpus, so the pattern is not in the data for the
model to pick up. We measure it every run rather than assuming.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

PATHOLOGIES = ["Cardiomegaly", "Edema", "Pleural_Effusion", "Atelectasis",
               "Consolidation", "Lung_Opacity", "Pneumonia", "Pneumothorax"]

# Short surface forms. These are the words radiologists actually write, which
# is what BioBART was pretrained on -- "Pleural_Effusion" is not.
SHORT_NAME = {"Cardiomegaly": "cardiomegaly", "Edema": "edema",
              "Pleural_Effusion": "pleural effusion", "Atelectasis": "atelectasis",
              "Consolidation": "consolidation", "Lung_Opacity": "opacity",
              "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax"}

PROMPT_DROPOUT = 0.15      # fraction of labels hidden from the prompt in training
MAX_PROMPT_TOKENS = 48


# ====================================================================
# 1 · prompt construction
# ====================================================================
def build_prompt(probs, thresholds, pathologies=PATHOLOGIES,
                 dropout: float = 0.0, rng=None) -> str:
    """Turn one image's classifier output into a text prompt.

    Both positives AND negatives are stated. Listing only positives would be
    shorter, but real reports assert negatives constantly ("no pneumothorax"),
    and a decoder told nothing about pneumothorax has no reason to mention it.

    `dropout` hides labels at random during training. Without it the decoder
    learns to transcribe the prompt and stops looking at the image, which
    collapses the moment the classifier is wrong.
    """
    rng = rng or np.random.default_rng()
    pos, neg = [], []
    for p, t, k in zip(probs, thresholds, pathologies):
        if dropout > 0 and rng.random() < dropout:
            continue                                   # label withheld
        (pos if p >= t else neg).append(SHORT_NAME[k])
    parts = []
    if pos:
        parts.append("positive: " + ", ".join(pos) + ".")
    if neg:
        parts.append("negative: " + ", ".join(neg) + ".")
    return " ".join(parts)


def encode_prompts(prompts, tokenizer, max_len: int = MAX_PROMPT_TOKENS,
                   device="cpu"):
    """Batch-tokenise prompts. Returns (ids, mask), both (B, L).

    add_special_tokens=False on purpose: BART will receive these as part of a
    longer encoder sequence, and injecting </s> mid-sequence would tell the
    encoder the input ended before the visual tokens even arrived.
    """
    enc = tokenizer(list(prompts), padding=True, truncation=True,
                    max_length=max_len, add_special_tokens=False,
                    return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ====================================================================
# 2 · model
# ====================================================================
class CXRConditionedGenerator(nn.Module):
    """Stage 4's generator, with a classifier-derived prompt prefix.

    Architecture is deliberately unchanged from Stage 4 -- `vision`, `proj` and
    `bart` keep the same names and shapes so `best.pt` loads by key. The only
    difference is what enters BART's encoder:

        Stage 4 :  [144 visual tokens]
        Stage 11:  [prompt tokens] ++ [144 visual tokens]

    Visual tokens still go in as inputs_embeds, so BART's pretrained encoder
    runs and adds its own positional embeddings. Passing them as
    encoder_outputs -- the original Stage 4 bug -- skips the encoder entirely
    and the decoder falls back on its language prior.
    """

    def __init__(self, decoder_name: str, proj_dropout: float = 0.1,
                 unfreeze_stages: int = 1):
        super().__init__()
        import torchvision
        from transformers import BartForConditionalGeneration
        base = torchvision.models.convnext_base(weights=None)
        self.vision = base.features
        for p in self.vision.parameters():
            p.requires_grad = False
        self.unfrozen = []
        if unfreeze_stages > 0:
            for m in list(self.vision.children())[-unfreeze_stages * 2:]:
                for p in m.parameters():
                    p.requires_grad = True
                self.unfrozen.append(m)

        self.bart = BartForConditionalGeneration.from_pretrained(decoder_name)
        d = self.bart.config.d_model
        self.proj = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU(),
            nn.Dropout(proj_dropout), nn.Linear(d, d), nn.LayerNorm(d))

    # ---------------- encoding ----------------
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.unfrozen:
            f = self.vision(images)
        else:
            with torch.no_grad():
                f = self.vision(images)
        f = f.flatten(2).transpose(1, 2)              # (B,144,1024)
        return self.proj(f)                            # (B,144,d)

    def embed_prompt(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed prompt ids with BART's own table, including its embed_scale.

        Omitting embed_scale is silent on bart-base (scale_embedding=False) but
        wrong on any checkpoint that sets it -- the prompt would arrive at a
        different magnitude from every other token the encoder has ever seen.
        """
        scale = (math.sqrt(self.bart.config.d_model)
                 if getattr(self.bart.config, "scale_embedding", False) else 1.0)
        return self.bart.model.shared(ids) * scale

    def build_inputs(self, images, prompt_ids=None, prompt_mask=None):
        """Returns (inputs_embeds, attention_mask) for BART's encoder."""
        vis = self.encode_image(images)
        vmask = torch.ones(vis.shape[:2], dtype=torch.long, device=vis.device)
        if prompt_ids is None or prompt_ids.numel() == 0:
            return vis, vmask                          # == Stage 4 exactly
        pe = self.embed_prompt(prompt_ids).to(vis.dtype)
        if prompt_mask is None:
            prompt_mask = torch.ones(pe.shape[:2], dtype=torch.long, device=pe.device)
        return (torch.cat([pe, vis], dim=1),
                torch.cat([prompt_mask.to(vmask.dtype), vmask], dim=1))

    def forward(self, images, labels, prompt_ids=None, prompt_mask=None):
        emb, mask = self.build_inputs(images, prompt_ids, prompt_mask)
        return self.bart(inputs_embeds=emb, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(self, images, prompt_ids=None, prompt_mask=None, **kw):
        """⚠️ READ BEFORE 'FIXING'.

        We run BART's encoder OURSELVES and hand generate() its OUTPUT. That is
        NOT the Stage 4 bug -- the bug was passing RAW projected ConvNeXt
        features as encoder_outputs so the encoder never ran. Here the encoder
        has already run; generate() simply must not run it a second time.
        """
        from transformers.modeling_outputs import BaseModelOutput
        emb, mask = self.build_inputs(images, prompt_ids, prompt_mask)
        enc = self.bart.model.encoder(inputs_embeds=emb, attention_mask=mask)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
            attention_mask=mask, **kw)

    # ---------------- checkpoint plumbing ----------------
    def load_stage4(self, ckpt: dict, use_ema: bool = True) -> dict:
        """Load the Stage 4 generator. Zero new parameters means this must be
        exact -- any missing or unexpected key is a real mismatch, not an
        expected consequence of adding a module.

        EMA MATTERS. Stage 4 selected its checkpoint on, and reported, the EMA
        weights. Loading only `model` silently starts from the raw weights,
        which are a different (worse) model than the 0.2918 ROUGE-L that was
        published -- and every Stage 11 gain would then be measured against
        the wrong baseline.
        """
        sd = ckpt.get("model", ckpt)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:8]}")
        if missing:
            raise RuntimeError(f"missing keys (expected none): {sorted(missing)[:8]}")
        applied = 0
        if use_ema and ckpt.get("ema"):
            msd = self.state_dict()
            bad = [k for k in ckpt["ema"] if k not in msd]
            if bad:
                raise RuntimeError(f"EMA keys absent from model: {bad[:5]}")
            for k, v in ckpt["ema"].items():
                msd[k].copy_(v.to(msd[k].dtype))
                applied += 1
        return dict(loaded=len(sd), missing=len(missing), ema_applied=applied)

    def param_groups(self, lr_vision: float, lr_rest: float):
        vis, rest = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (vis if n.startswith("vision") else rest).append(p)
        return [{"params": vis, "lr": lr_vision}, {"params": rest, "lr": lr_rest}]


# ====================================================================
# 3 · self-tests
# ====================================================================
def _selftest(verbose: bool = True) -> tuple[int, int]:
    P, F = [], []

    def g(name, ok, extra=""):
        (P if ok else F).append(name)
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{extra}")

    rng = np.random.default_rng(0)
    thr = [0.5] * 8
    probs = [0.9, 0.1, 0.8, 0.2, 0.05, 0.3, 0.02, 0.01]

    s = build_prompt(probs, thr)
    g("prompt lists positives", "cardiomegaly" in s and "pleural effusion" in s, s[:58])
    g("prompt lists negatives", "negative:" in s and "pneumothorax" in s)
    g("positives come before negatives", s.index("positive:") < s.index("negative:"))
    g("uses radiologist wording, not column names",
      "Pleural_Effusion" not in s and "pleural effusion" in s)
    g("all-negative case omits the positive clause",
      not build_prompt([0.0] * 8, thr).startswith("positive"))
    g("all-positive case omits the negative clause",
      "negative" not in build_prompt([1.0] * 8, thr))

    n_full = len(build_prompt(probs, thr).split(","))
    drops = [len(build_prompt(probs, thr, dropout=0.5,
                              rng=np.random.default_rng(i)).split(","))
             for i in range(40)]
    g("label dropout removes labels", min(drops) < n_full,
      f"full={n_full} min_dropped={min(drops)}")
    g("dropout=0 is deterministic",
      build_prompt(probs, thr, dropout=0.0) == build_prompt(probs, thr, dropout=0.0))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/bart-base")
    ids, mask = encode_prompts([s, "positive: edema."], tok)
    g("prompts batch-tokenise and pad", ids.shape[0] == 2 and ids.shape == mask.shape,
      str(tuple(ids.shape)))
    g("no special tokens injected mid-sequence",
      tok.eos_token_id not in ids[0].tolist())
    g("padding is masked out", int(mask[1].sum()) < int(mask[0].sum()))

    m = CXRConditionedGenerator("facebook/bart-base", unfreeze_stages=1).eval()
    xb = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        v_only, m_only = m.build_inputs(xb)
        v_pr, m_pr = m.build_inputs(xb, ids, mask)
    g("no prompt -> visual tokens only (Stage 4 behaviour)",
      v_only.shape[1] == v_pr.shape[1] - ids.shape[1], str(tuple(v_only.shape)))
    g("*** empty prompt is BIT-IDENTICAL to Stage 4 ***",
      torch.equal(v_pr[:, ids.shape[1]:], v_only),
      "visual half unchanged by prepending")
    g("mask length matches embedding length",
      m_pr.shape[1] == v_pr.shape[1] and m_only.shape == v_only.shape[:2])
    g("prompt padding stays masked in the combined mask",
      int(m_pr[1].sum()) < int(m_pr[0].sum()))

    lab = torch.randint(4, 900, (2, 12))
    with torch.no_grad():
        o1 = m(xb, lab, ids, mask)
        o2 = m(xb, lab)
    g("forward with prompt returns finite loss",
      torch.isfinite(o1.loss).item(), f"{o1.loss.item():.4f}")
    g("prompt CHANGES the loss (it is being used)",
      not torch.allclose(o1.loss, o2.loss), f"{o2.loss.item():.4f} vs {o1.loss.item():.4f}")

    with torch.no_grad():
        gA = m.generate(xb, ids, mask, num_beams=1, max_length=20, min_length=5)
        gB = m.generate(xb, num_beams=1, max_length=20, min_length=5)
    g("generation works with a prompt", tuple(gA.shape)[0] == 2, str(tuple(gA.shape)))
    g("generation works without a prompt", tuple(gB.shape)[0] == 2)
    ids2, mask2 = encode_prompts(["positive: pneumothorax.", "negative: edema."], tok)
    with torch.no_grad():
        gC = m.generate(xb, ids2, mask2, num_beams=1, max_length=20, min_length=5)
    L = min(gA.shape[1], gC.shape[1])
    g("*** a DIFFERENT prompt gives a different report ***",
      not torch.equal(gA[:, :L], gC[:, :L]), "the prompt actually steers generation")

    g("embed_scale honoured",
      abs(float(m.embed_prompt(ids).abs().mean())
          - float((m.bart.model.shared(ids)).abs().mean()
                  * (math.sqrt(m.bart.config.d_model)
                     if getattr(m.bart.config, "scale_embedding", False) else 1.0))) < 1e-5)

    ref = CXRConditionedGenerator("facebook/bart-base", unfreeze_stages=1)
    info = m.load_stage4({"model": ref.state_dict()})
    g("loads a Stage-4-shaped checkpoint with zero new keys",
      info["loaded"] > 300, str(info))
    try:
        m.load_stage4({"model": {"bogus.key": torch.zeros(1)}}); ok = False
    except RuntimeError:
        ok = True
    g("rejects a mismatched checkpoint (fails loud)", ok)
    g("param_groups separates vision from the rest",
      len(m.param_groups(1e-6, 1e-5)) == 2)

    if verbose:
        print(f"\n  {len(P)} passed, {len(F)} failed")
        for f in F:
            print(f"    - {f}")
    return len(P), len(F)


if __name__ == "__main__":
    print("=" * 80)
    print(" STAGE 11 · stage11_conditioned.py self-test")
    print("=" * 80)
    p, f = _selftest()
    raise SystemExit(1 if f else 0)
