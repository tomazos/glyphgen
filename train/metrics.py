from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    m = mask.to(a.dtype).unsqueeze(-1)
    loss = (m * (a - b).abs()).sum() / m.sum().clamp_min(eps)
    return loss


def closure_error(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean L2 distance between last valid on-curve point and the first point per contour.
    Here we approximate by using the last valid point in each contour (mask) vs first point.
    Returns scalar.
    """
    # points: [B,K,L,2], mask: [B,K,L]
    B, K, L, _ = points.shape
    first = points[:, :, :1, :]  # [B,K,1,2]
    # index of last valid per contour
    lengths = mask.long().sum(dim=2).clamp_min(1)  # [B,K]
    idx = (lengths - 1).view(B, K, 1, 1).expand(B, K, 1, 2)
    last = torch.gather(points, 2, idx)
    d = ((last - first) ** 2).sum(dim=-1).sqrt()  # [B,K,1]
    # average over contours that have >=1 point
    mK = (lengths > 0).float().view(B, K, 1)
    return (d * mK).sum() / mK.sum().clamp_min(1.0)


def chamfer_distance(a: torch.Tensor, b: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between two point sets per contour, averaged over batch.
    a,b: [B,K,L,2]; masks: [B,K,L]
    Computes a simplified batched version by flattening contours; good enough for early training.
    """
    # Flatten valid points across K,L
    def flatten(p, m):
        B, K, L, _ = p.shape
        m = m.unsqueeze(-1).to(p.dtype)
        p = p * m
        # pack to [B, N, 2]
        p = p.view(B, K * L, 2)
        m = m.view(B, K * L, 1)
        return p, m

    a, ma = flatten(a, mask_a)
    b, mb = flatten(b, mask_b)

    # For numerical stability, mask out invalid points by setting them far away and zeroing their contributions
    BIG = 1e6
    a_masked = torch.where(ma.bool(), a, torch.full_like(a, BIG))
    b_masked = torch.where(mb.bool(), b, torch.full_like(b, -BIG))

    # pairwise distances: [B, Na, Nb]
    # (a-b)^2 = a^2 + b^2 - 2ab
    a2 = (a_masked ** 2).sum(-1, keepdim=True)  # [B, Na, 1]
    b2 = (b_masked ** 2).sum(-1).unsqueeze(1)   # [B, 1, Nb]
    ab = a_masked @ b_masked.transpose(1, 2)    # [B, Na, Nb]
    d2 = a2 + b2 - 2 * ab

    # For invalid rows/cols, distances are huge; min will ignore them effectively
    da, _ = d2.min(dim=2)  # [B, Na]
    db, _ = d2.min(dim=1)  # [B, Nb]

    # Average only over valid points
    ca = (da * ma.squeeze(-1)).sum(dim=1) / ma.squeeze(-1).sum(dim=1).clamp_min(1.0)
    cb = (db * mb.squeeze(-1)).sum(dim=1) / mb.squeeze(-1).sum(dim=1).clamp_min(1.0)
    return ((ca + cb) * 0.5).mean()
