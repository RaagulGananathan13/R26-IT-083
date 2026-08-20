"""
Report generator — Stage 11 (falls back to Stage 4).

    ConvNeXt features (B,1024,12,12)
      -> 144 visual tokens -> MLP projection -> (B,144,768)
      -> BART ENCODER            <- via inputs_embeds
      -> BART decoder, greedy    -> "FINDINGS: ... IMPRESSION: ..."

WHY inputs_embeds AND NOT encoder_outputs
-----------------------------------------
The original deployed model (models/report_generator/best_model.pt, April 2026)
passed projected visual features as `encoder_outputs`, which bypasses BART's
pretrained encoder entirely. The decoder's cross-attention was pretrained to
read encoder OUTPUTS of a particular scale and structure; handed raw projected
convolutional features it cannot use them and falls back on its language prior.
That model scored ROUGE-L 0.2740 -- BELOW the 0.2769 constant-string baseline --
and fabricated a reference to a non-existent prior study in ~63% of reports.

Using inputs_embeds means BART's encoder actually runs and adds its own learned
positional embeddings, so the decoder can distinguish apex from base.

Stage 11 additionally prepends a classifier-derived text prompt. The ablation
showed the prompt itself contributes only +0.0023 clinical F1 (the +0.0138 gain
came from the extra fine-tuning), so the prompt is retained for fidelity to the
trained checkpoint rather than claimed as the source of the improvement.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class CXRReportGenerator(nn.Module):
    """Matches Stage 4/Stage 11 checkpoints: `vision`, `proj`, `bart`."""

    def __init__(self, decoder_name: str, proj_dropout: float = 0.1):
        super().__init__()
        from torchvision import models
        from transformers import BartForConditionalGeneration
        base = models.convnext_base(weights=None)
        self.vision = base.features
        self.bart = BartForConditionalGeneration.from_pretrained(decoder_name)
        d = self.bart.config.d_model
        self.proj = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, d), nn.GELU(),
            nn.Dropout(proj_dropout), nn.Linear(d, d), nn.LayerNorm(d))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        f = self.vision(images)                       # (B,1024,12,12)
        f = f.flatten(2).transpose(1, 2)              # (B,144,1024)
        return self.proj(f)                           # (B,144,d)

    def embed_prompt(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed with BART's own table, including embed_scale.

        Omitting embed_scale is silent on bart-base (scale_embedding=False) but
        wrong on any checkpoint that sets it -- the prompt would reach the
        encoder at a different magnitude from every other token.
        """
        scale = (math.sqrt(self.bart.config.d_model)
                 if getattr(self.bart.config, "scale_embedding", False) else 1.0)
        return self.bart.model.shared(ids) * scale

    def build_inputs(self, images, prompt_ids=None, prompt_mask=None):
        vis = self.encode_image(images)
        vmask = torch.ones(vis.shape[:2], dtype=torch.long, device=vis.device)
        if prompt_ids is None or prompt_ids.numel() == 0:
            return vis, vmask
        pe = self.embed_prompt(prompt_ids).to(vis.dtype)
        if prompt_mask is None:
            prompt_mask = torch.ones(pe.shape[:2], dtype=torch.long, device=pe.device)
        return (torch.cat([pe, vis], dim=1),
                torch.cat([prompt_mask.to(vmask.dtype), vmask], dim=1))

    @torch.no_grad()
    def generate(self, images, prompt_ids=None, prompt_mask=None, **kw):
        """We run BART's encoder ourselves and hand generate() its OUTPUT.

        This is NOT the old bug. The bug passed RAW projected features as
        encoder_outputs so the encoder never ran. Here the encoder has already
        run; generate() simply must not run it a second time.
        """
        from transformers.modeling_outputs import BaseModelOutput
        emb, mask = self.build_inputs(images, prompt_ids, prompt_mask)
        enc = self.bart.model.encoder(inputs_embeds=emb, attention_mask=mask)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
            attention_mask=mask, **kw)


def load_report_generator(weights_path, decoder_name, device):
    """Load Stage 11 or Stage 4. Returns (model, tokenizer, info)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(decoder_name)
    m = CXRReportGenerator(decoder_name)

    ck = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError("unexpected checkpoint keys: %s" % unexpected[:6])
    if missing:
        raise RuntimeError("missing keys: %s" % sorted(missing)[:6])

    n_ema = 0
    if ck.get("ema"):
        msd = m.state_dict()
        for k, v in ck["ema"].items():
            if k in msd:
                msd[k].copy_(v.to(msd[k].dtype))
                n_ema += 1

    info = dict(epoch=ck.get("epoch"), metric=ck.get("best_metric"), ema=n_ema)
    print("[report_gen] epoch %s | metric %s | EMA applied: %d"
          % (info["epoch"], info["metric"], n_ema))
    return m.to(device).eval(), tok, info
