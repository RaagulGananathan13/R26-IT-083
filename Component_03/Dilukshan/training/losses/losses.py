"""
UEF-Net multi-task objective.

Components
----------
1. Regression : LDS-weighted Huber on standardized EF                (-> MAE)
2. Ordinal    : soft-CORAL BCE with MEASUREMENT-UNCERTAINTY soft labels.
                NOVELTY: EF is itself a noisy measurement (inter-observer
                sigma ~= 4 EF pts).  Instead of hard class labels we set the
                cumulative target for threshold t_k to
                    s_k = P(trueEF > t_k | measuredEF) = 1 - Phi((t_k - EF)/sigma)
                so boundary cases carry honest, soft supervision.       (-> balanced cls)
3. Consistency: the regression head and the ordinal head must agree on the
                cumulative probabilities P(y>t_k).                       (-> coupling)

Total = w_reg*L_reg + w_ord*L_ord + w_consistency*L_con
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ndtr(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def soft_cumulative_targets(ef: torch.Tensor, thresholds: torch.Tensor,
                            sigma: float) -> torch.Tensor:
    """s_k = P(trueEF > t_k | measuredEF) -> (B, K-1)."""
    ef = ef.unsqueeze(1)                        # (B,1)
    thr = thresholds.unsqueeze(0)              # (1,K-1)
    z = (thr - ef) / max(sigma, 1e-6)
    return (1.0 - _ndtr(z)).clamp(1e-4, 1 - 1e-4)


def reg_cumulative_probs(ef_hat: torch.Tensor, thresholds: torch.Tensor,
                         sigma: float) -> torch.Tensor:
    """Cumulative P(y>t_k) implied by the regressed EF (for consistency)."""
    ef_hat = ef_hat.unsqueeze(1)
    thr = thresholds.unsqueeze(0)
    z = (thr - ef_hat) / max(sigma, 1e-6)
    return (1.0 - _ndtr(z)).clamp(1e-4, 1 - 1e-4)


def soft_class_targets(ef: torch.Tensor, thresholds: torch.Tensor,
                       sigma: float) -> torch.Tensor:
    """Gaussian measurement-noise target mass for each clinical EF interval."""
    ef = ef.unsqueeze(1)
    z = (thresholds.unsqueeze(0) - ef) / max(sigma, 1e-6)
    cdf = _ndtr(z).clamp(1e-5, 1.0 - 1e-5)
    pieces = [cdf[:, :1]]
    if cdf.shape[1] > 1:
        pieces.append(cdf[:, 1:] - cdf[:, :-1])
    pieces.append(1.0 - cdf[:, -1:])
    out = torch.cat(pieces, dim=1).clamp_min(1e-6)
    return out / out.sum(dim=1, keepdim=True)


def ordinal_class_distribution(ord_logits: torch.Tensor) -> torch.Tensor:
    """Convert monotone cumulative probabilities into K class probabilities."""
    p = torch.sigmoid(ord_logits)
    one = torch.ones_like(p[:, :1])
    zero = torch.zeros_like(p[:, :1])
    dist = torch.cat([one, p], 1) - torch.cat([p, zero], 1)
    dist = dist.clamp_min(0.0)
    return dist / dist.sum(dim=1, keepdim=True).clamp_min(1e-6)


def pairwise_rank_loss(ef_hat: torch.Tensor, ef: torch.Tensor,
                       min_gap: float = 3.0, temperature: float = 4.0) -> torch.Tensor:
    """Smooth pairwise ordering loss; ignores clinically indistinguishable pairs."""
    target_delta = ef[:, None] - ef[None, :]
    pred_delta = ef_hat[:, None] - ef_hat[None, :]
    mask = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
    mask &= target_delta.abs() >= min_gap
    if not bool(mask.any()):
        return ef_hat.new_zeros(())
    signed_margin = target_delta.sign() * pred_delta / max(temperature, 1e-6)
    return F.softplus(-signed_margin[mask]).mean()


class UEFLoss(nn.Module):
    def __init__(self, cfg, ef_mean: float, ef_std: float, class_counts=None):
        super().__init__()
        self.cfg = cfg
        self.ef_mean = ef_mean
        self.ef_std = ef_std
        self.sigma = cfg.ef_noise_sigma
        self.class_sigma = float(getattr(cfg, "class_target_sigma", self.sigma))
        self.register_buffer("thresholds",
                             torch.tensor(cfg.EF_THRESHOLDS, dtype=torch.float32))
        if class_counts is None:
            class_counts = torch.ones(cfg.n_classes, dtype=torch.float32)
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
        beta = float(getattr(cfg, "effective_num_beta", 0.9999))
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective.clamp_min(1e-12)
        self.register_buffer("class_weights", weights / weights.mean())
        # Logit-adjustment prior (Menon et al. 2021): log P(class) over the
        # TRAIN distribution.  Added to the class-head logits when tau > 0.
        self.la_tau = float(getattr(cfg, "logit_adjustment_tau", 0.0))
        log_prior = torch.log((counts / counts.sum()).clamp_min(1e-8))
        self.register_buffer("log_prior", log_prior)

    def forward(self, ef_z_pred, ord_logits, batch, aux=None, epoch: int = 0):
        ef = batch["ef"]                        # raw EF (B,)
        ef_z = batch["ef_z"]                    # standardized target
        lds_w = batch["lds_w"]                  # (B,)

        # 1) regression (LDS-weighted Huber on standardized EF)
        reg = F.huber_loss(ef_z_pred, ef_z, delta=self.cfg.huber_delta, reduction="none")
        lds_power = float(getattr(self.cfg, "lds_weight_power", 1.0))
        effective_lds = lds_w.clamp_min(1e-6).pow(lds_power)
        if getattr(self.cfg, "model_version", "uefnet_v1") != "uefnet_v1":
            # Preserve the relative tail emphasis without changing the effective
            # regression learning rate from one mini-batch to another.
            effective_lds = effective_lds / effective_lds.mean().clamp_min(1e-6)
        L_reg = (reg * effective_lds).mean()

        # 2) soft-CORAL ordinal with uncertainty soft labels
        soft_t = soft_cumulative_targets(ef, self.thresholds, self.sigma)   # (B,K-1)
        ord_each = F.binary_cross_entropy_with_logits(
            ord_logits, soft_t, reduction="none").mean(dim=1)

        model_version = getattr(self.cfg, "model_version", "uefnet_v1")
        if model_version == "uefnet_v1":
            L_ord = ord_each.mean()
            class_each = ef_z_pred.new_zeros(ef_z_pred.shape)
            L_class = ef_z_pred.new_zeros(())
            L_nll = ef_z_pred.new_zeros(())
            L_rank = ef_z_pred.new_zeros(())
            L_heads = ef_z_pred.new_zeros(())
        else:
            y_class = batch["ef_class"]
            use_deferred = epoch >= int(getattr(self.cfg, "drw_epoch", 0))
            cls_w = self.class_weights[y_class] if use_deferred else torch.ones_like(ord_each)
            # Normalise within a batch so DRW changes relative emphasis, not LR scale.
            cls_w = cls_w / cls_w.mean().clamp_min(1e-6)
            L_ord = (ord_each * cls_w).mean()

            aux = aux if isinstance(aux, dict) else {}
            class_logits = aux.get("class_logits")
            target_dist = soft_class_targets(ef, self.thresholds, self.class_sigma)
            if class_logits is not None:
                # Logit adjustment: shift logits by tau*log(prior) so rare/middle
                # classes need a larger raw margin to be predicted (Menon 2021).
                # At inference plain argmax is used (no shift), so this only
                # changes the training objective, not the exported probabilities.
                adj_logits = (class_logits + self.la_tau * self.log_prior.unsqueeze(0)
                              if self.la_tau > 0.0 else class_logits)
                class_each = -(target_dist * F.log_softmax(adj_logits, dim=1)).sum(dim=1)
                L_class = (class_each * cls_w).mean()
                L_heads = F.mse_loss(
                    F.softmax(class_logits, dim=1),
                    ordinal_class_distribution(ord_logits), reduction="mean")
            else:
                class_each = ef_z_pred.new_zeros(ef_z_pred.shape)
                L_class = ef_z_pred.new_zeros(())
                L_heads = ef_z_pred.new_zeros(())

            log_var = aux.get("log_var")
            if log_var is not None:
                sq = (ef_z_pred - ef_z).pow(2)
                L_nll = (0.5 * (torch.exp(-log_var) * sq + log_var) * effective_lds).mean()
            else:
                L_nll = ef_z_pred.new_zeros(())

            ef_hat_rank = ef_z_pred * self.ef_std + self.ef_mean
            L_rank = pairwise_rank_loss(
                ef_hat_rank, ef,
                min_gap=float(getattr(self.cfg, "rank_min_gap", 3.0)),
                temperature=float(getattr(self.cfg, "rank_temperature", 4.0)))

        # 3) dual-head consistency
        ef_hat = ef_z_pred * self.ef_std + self.ef_mean
        p_reg = reg_cumulative_probs(ef_hat, self.thresholds, self.sigma)   # (B,K-1)
        p_ord = torch.sigmoid(ord_logits)
        L_con = F.mse_loss(p_reg, p_ord, reduction="mean")

        total = (self.cfg.w_reg * L_reg + self.cfg.w_ord * L_ord
                 + self.cfg.w_consistency * L_con
                 + float(getattr(self.cfg, "w_class", 0.0)) * L_class
                 + float(getattr(self.cfg, "w_nll", 0.0)) * L_nll
                 + float(getattr(self.cfg, "w_rank", 0.0)) * L_rank
                 + float(getattr(self.cfg, "w_head_consistency", 0.0)) * L_heads)
        return total, {"loss": total.item(), "reg": L_reg.item(),
                       "ord": L_ord.item(), "con": L_con.item(),
                       "class": L_class.item(), "nll": L_nll.item(),
                       "rank": L_rank.item(), "heads": L_heads.item()}
