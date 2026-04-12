"""
Loss functions for instance-segmentation training.

DeepClusteringLoss is the affinity formulation from Hershey et al. (2016),
"Deep Clustering: Discriminative Embeddings for Segmentation and Separation".
The idea in one sentence: embed each T-F bin in a D-dim unit sphere, then
train so that bins from the same source have high dot product and bins from
different sources have low dot product.

InstanceSegLoss bundles the DC embedding loss with a standard BCE on the
rumble-vs-noise mask so that a single backward pass updates both heads.

Both losses expect a binary or soft `active_mask` so that "silence / non-rumble"
bins (which don't belong to any source) are excluded from the affinity term.
Without this the network wastes capacity trying to cluster background noise
bins, and the DC loss dominates with nonsense.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DeepClusteringLoss(nn.Module):
    """Affinity-based embedding loss.

    Given
        V : [B, D, F, T]  L2-normalised per-bin embeddings (from UNet instance head)
        Y : [B, K, F, T]  one-hot or soft per-bin source assignments (labels)
        M : [B, 1, F, T]  active mask - bins the network should care about

    Reshape each to [B, N, *] where N = F*T is the bin count. The loss is

        L = ||V V^T - Y Y^T||_F^2

    restricted to active bins. Equivalent to forcing the embedding affinity
    matrix V V^T to match the ground-truth affinity Y Y^T. We expand the
    Frobenius-norm expression into the three trace terms below so we never
    have to materialise the N x N matrix.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor,          # [B, D, F, T]
        labels: torch.Tensor,              # [B, K, F, T]
        active_mask: Optional[torch.Tensor] = None,  # [B, 1, F, T]
    ) -> torch.Tensor:
        if embeddings.dim() != 4 or labels.dim() != 4:
            raise ValueError("embeddings and labels must be [B, C, F, T]")
        B, D, F, T = embeddings.shape
        K = labels.shape[1]
        N = F * T

        V = embeddings.reshape(B, D, N).transpose(1, 2)   # [B, N, D]
        Y = labels.reshape(B, K, N).transpose(1, 2).float()  # [B, N, K]

        if active_mask is not None:
            M = active_mask.reshape(B, N, 1).float()  # [B, N, 1]
            V = V * M
            Y = Y * M

        # Expand ||VV^T - YY^T||^2 so we never form the N x N affinity matrix:
        #   = tr(VV^T VV^T) - 2 tr(VV^T YY^T) + tr(YY^T YY^T)
        #   = ||V^T V||^2   - 2 ||V^T Y||^2   + ||Y^T Y||^2
        # (all matrices here are the small D x D / D x K / K x K variants)
        VtV = torch.bmm(V.transpose(1, 2), V)  # [B, D, D]
        VtY = torch.bmm(V.transpose(1, 2), Y)  # [B, D, K]
        YtY = torch.bmm(Y.transpose(1, 2), Y)  # [B, K, K]

        vv = (VtV ** 2).flatten(1).sum(dim=1)
        vy = (VtY ** 2).flatten(1).sum(dim=1)
        yy = (YtY ** 2).flatten(1).sum(dim=1)

        loss = vv - 2.0 * vy + yy
        # Normalise by the active-bin affinity norm so loss magnitude is
        # stable regardless of how much of the chunk contains rumble.
        norm = vv + yy + self.eps
        return (loss / norm).mean()


class InstanceSegLoss(nn.Module):
    """Combined mask (BCE-with-logits) + Deep Clustering embedding loss."""

    def __init__(self, dc_weight: float = 1.0, mask_weight: float = 1.0) -> None:
        super().__init__()
        self.mask_loss = nn.BCEWithLogitsLoss()
        self.dc_loss = DeepClusteringLoss()
        self.dc_weight = dc_weight
        self.mask_weight = mask_weight

    def forward(
        self,
        mask_logits: torch.Tensor,    # [B, 1, F, T]
        embeddings: torch.Tensor,     # [B, D, F, T]
        mask_target: torch.Tensor,    # [B, 1, F, T] binary rumble/noise mask
        instance_labels: torch.Tensor,  # [B, K, F, T] per-source one-hot
    ) -> dict:
        m_loss = self.mask_loss(mask_logits, mask_target)
        # Use the ground-truth rumble mask to gate the DC loss: embeddings
        # for non-rumble bins don't need to cluster at all.
        dc_loss = self.dc_loss(embeddings, instance_labels, active_mask=mask_target)
        total = self.mask_weight * m_loss + self.dc_weight * dc_loss
        return {"total": total, "mask": m_loss.detach(), "dc": dc_loss.detach()}


if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, K, F, T = 2, 20, 2, 64, 32
    emb = torch.randn(B, D, F, T)
    emb = emb / emb.norm(dim=1, keepdim=True)   # pretend the UNet normalised

    # Fake ground-truth assignments: half the bins -> source 0, half -> source 1.
    labels = torch.zeros(B, K, F, T)
    labels[:, 0, : F // 2, :] = 1.0
    labels[:, 1, F // 2 :, :] = 1.0

    mask_logits = torch.randn(B, 1, F, T)
    mask_target = torch.ones(B, 1, F, T)

    dc = DeepClusteringLoss()
    iseg = InstanceSegLoss(dc_weight=1.0, mask_weight=1.0)

    print(f"[losses] DC loss (random emb)   : {dc(emb, labels, mask_target):.4f}")
    print(f"[losses] InstanceSegLoss bundle : "
          f"{iseg(mask_logits, emb, mask_target, labels)}")
    print("[losses] OK - both losses produce finite scalars.")
