"""
The report generator: turns an X-ray into a written radiology report.

The flow is:

    ConvNeXt features (B, 1024, 12, 12)
      -> flatten the 12x12 grid into 144 "visual tokens"
      -> project them from 1024 dims to BART's 768
      -> BART encoder            <- passed as inputs_embeds
      -> BART decoder, greedy    -> "FINDINGS: ... IMPRESSION: ..."

WHY inputs_embeds AND NOT encoder_outputs

This is the single most important detail in the file. An earlier version handed
the projected image features straight in as `encoder_outputs`, which skips
BART's encoder completely. The decoder's cross-attention was pretrained to read
encoder OUTPUTS, which have a particular scale and structure. Give it raw
convolutional features instead and it simply cannot use them, so it falls back
on what it remembers about how radiology reports usually sound.

The result was a model that wrote fluent, believable reports without really
looking at the X-ray. It scored ROUGE-L 0.2740, which is BELOW the 0.2769 you
get from printing the same fixed paragraph for every patient, and it invented a
reference to a prior study in about 63% of reports.

Going through inputs_embeds means the encoder actually runs and adds its own
positional embeddings, so the decoder can tell the top of the lung from the
bottom.

Stage 11 also puts a short text prompt in front, built from the classifier's
predictions. Being honest about this: the ablation showed the prompt itself is
worth only +0.0023 clinical F1. Most of the +0.0138 gain came from the extra
fine-tuning. We keep the prompt because the checkpoint was trained with it, not
because it is doing the work.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class CXRReportGenerator(nn.Module):
    """Matches the saved Stage 4 / Stage 11 checkpoints: vision, proj, bart."""

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
        # 12x12 grid -> a row of 144 tokens -> translated into BART's dimension
        f = self.vision(images)                       # (B,1024,12,12)
        f = f.flatten(2).transpose(1, 2)              # (B,144,1024)
        return self.proj(f)                           # (B,144,d)

    def embed_prompt(self, ids: torch.Tensor) -> torch.Tensor:
        """Turn prompt token ids into embeddings using BART's own table.

        The embed_scale bit matters. On bart-base it is off, so forgetting it
        changes nothing and you never notice. On a checkpoint that turns it on,
        the prompt would arrive at a different magnitude from every other token.
        """
        scale = (math.sqrt(self.bart.config.d_model)
                 if getattr(self.bart.config, "scale_embedding", False) else 1.0)
        return self.bart.model.shared(ids) * scale

    def build_inputs(self, images, prompt_ids=None, prompt_mask=None):
        # Prompt tokens first (if any), then the 144 image tokens.
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
        """Run the encoder ourselves, then let generate() decode from its output.

        To be clear, this is NOT the old bug. The bug passed raw projected
        features as encoder_outputs so the encoder never ran at all. Here the
        encoder has already run on the line above; we pass its result so
        generate() doesn't run it a second time.
        """
        from transformers.modeling_outputs import BaseModelOutput
        emb, mask = self.build_inputs(images, prompt_ids, prompt_mask)
        enc = self.bart.model.encoder(inputs_embeds=emb, attention_mask=mask)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
            attention_mask=mask, **kw)


def load_report_generator(weights_path, decoder_name, device):
    """Load Stage 11 (or Stage 4). Returns (model, tokenizer, info)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(decoder_name)
    m = CXRReportGenerator(decoder_name)

    ck = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    # Fail loudly on any mismatch. A silently half-loaded model still produces
    # text, it is just nonsense text.
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
