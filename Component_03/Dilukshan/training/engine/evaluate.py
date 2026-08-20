"""
Video-level inference with multi-clip test-time augmentation (TTA).

Returns per-video predictions:
  ef_true, ef_pred (averaged over clips),
  y_true, ord_dist (averaged ordinal class distribution), ord_pred (argmax).
For uefnet_v2 the auxiliary softmax classification head is additionally exposed
as class_dist (averaged softmax) / class_pred (argmax), so its calibrated
decision strategies are usable at evaluation time.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.uef_net import coral_class_distribution


@torch.no_grad()
def run_inference(model, loader, cfg, device, ef_mean: float, ef_std: float,
                  max_batches: int = 0, desc: str = None) -> dict:
    model.eval()
    ef_true, ef_pred, y_true = [], [], []
    ef_pred_std, ord_dists, class_dists, aleatoric_stds = [], [], [], []

    use_amp = cfg.amp and device.type == "cuda"
    total = max_batches if max_batches else len(loader)
    iterator = enumerate(loader)
    if desc is not None:
        iterator = enumerate(tqdm(loader, total=total, desc=desc, leave=False))
    for bi, batch in iterator:
        if max_batches and bi >= max_batches:
            break
        vid = batch["video"].to(device, non_blocking=True)
        multiview = (vid.dim() == 6)                     # (B,V,C,T,H,W)
        if multiview:
            B, V = vid.shape[0], vid.shape[1]
            vid = vid.reshape(B * V, *vid.shape[2:])
        elif vid.dim() != 5:
            raise ValueError(f"expected video tensor (B,C,T,H,W) or (B,V,C,T,H,W), got {vid.shape}")

        # Flattening many TTA clips can turn a safe evaluation batch into a
        # large effective batch (for example 4 videos x 10 views).  Forward in
        # bounded chunks and then aggregate exactly at video level.
        forward_batch = max(1, int(getattr(cfg, "tta_forward_batch", cfg.batch_size)))
        z_chunks, dist_chunks, class_chunks, var_chunks = [], [], [], []
        for start in range(0, len(vid), forward_batch):
            chunk = vid[start:start + forward_batch]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                z, logits, aux = model(chunk)
            z_chunks.append(z.float())
            dist_chunks.append(coral_class_distribution(logits.float()))
            # v2 forward returns an aux dict; v1 returns a feature tensor.
            if isinstance(aux, dict) and aux.get("class_logits") is not None:
                class_chunks.append(F.softmax(aux["class_logits"].float(), dim=1))
            # Learned (aleatoric) predictive variance from the log-variance head,
            # trained by the Gaussian NLL term.  Exposing it enables selective
            # prediction; it is optional so v1 behaviour is unchanged.
            if isinstance(aux, dict) and aux.get("log_var") is not None:
                var_chunks.append(torch.exp(aux["log_var"].float()))
        ef_z = torch.cat(z_chunks, dim=0)
        dist = torch.cat(dist_chunks, dim=0)                   # (N,K)
        class_dist = torch.cat(class_chunks, dim=0) if class_chunks else None
        aleatoric_var = torch.cat(var_chunks, dim=0) if var_chunks else None

        if multiview:
            ef_views = ef_z.view(B, V)
            view_std = ef_views.std(dim=1, unbiased=False)
            ef_z = ef_views.mean(dim=1)
            dist = dist.view(B, V, -1).mean(dim=1)
            if class_dist is not None:
                class_dist = class_dist.view(B, V, -1).mean(dim=1)
            if aleatoric_var is not None:
                # Law of total variance: average the per-view aleatoric variance.
                aleatoric_var = aleatoric_var.view(B, V).mean(dim=1)
        else:
            view_std = torch.zeros_like(ef_z)

        if not torch.isfinite(ef_z).all() or not torch.isfinite(dist).all():
            raise FloatingPointError(f"non-finite prediction encountered in inference batch {bi}")

        ef_hat = ef_z.cpu().numpy() * ef_std + ef_mean
        ef_pred.append(ef_hat)
        ef_pred_std.append(view_std.cpu().numpy() * abs(float(ef_std)))
        ord_dists.append(dist.cpu().numpy())
        if class_dist is not None:
            class_dists.append(class_dist.cpu().numpy())
        if aleatoric_var is not None:
            # standardized variance -> EF units
            aleatoric_stds.append(
                np.sqrt(aleatoric_var.cpu().numpy()) * abs(float(ef_std)))
        ef_true.append(batch["ef"].detach().cpu().numpy())
        y_true.append(batch["ef_class"].detach().cpu().numpy())

    if not ef_pred:
        raise RuntimeError("inference loader produced no batches")
    ef_true = np.concatenate(ef_true)
    ef_pred = np.clip(np.concatenate(ef_pred), 0.0, 100.0)   # EF is physically in (0,100]
    ef_pred_std = np.concatenate(ef_pred_std)
    y_true = np.concatenate(y_true).astype(np.int64)
    ord_dist = np.concatenate(ord_dists, axis=0)
    ord_pred = ord_dist.argmax(axis=1).astype(np.int64)
    out = dict(ef_true=ef_true, ef_pred=ef_pred, ef_pred_std=ef_pred_std, y_true=y_true,
               ord_dist=ord_dist, ord_pred=ord_pred)
    # Only present for uefnet_v2 (auxiliary softmax head); consumers must treat
    # class_dist/class_pred as optional so v1 evaluation is unchanged.
    if len(class_dists) == len(ord_dists) and class_dists:
        class_dist = np.concatenate(class_dists, axis=0)
        out["class_dist"] = class_dist
        out["class_pred"] = class_dist.argmax(axis=1).astype(np.int64)
    # Learned aleatoric std in EF units (uefnet_v2 only), for selective prediction.
    if len(aleatoric_stds) == len(ord_dists) and aleatoric_stds:
        out["ef_aleatoric_std"] = np.concatenate(aleatoric_stds)
    return out
