"""
Exponential Moving Average of model weights (Polyak averaging).

Maintains a shadow copy whose parameters/buffers track the training model with
a slow decay.  Evaluating and checkpointing the EMA weights instead of the raw
weights is a well-established, low-risk generalisation booster — especially
useful here where the per-epoch min-recall is noisy, because EMA effectively
averages over the last ~1/(1-decay) optimiser steps.
"""
from __future__ import annotations
import copy
import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach().to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(mv)                       # ints (e.g. num_batches_tracked)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
