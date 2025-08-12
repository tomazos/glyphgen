from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """Mean over `dim` with boolean mask. Keeps reduced dim for broadcasting.
    x: (..., N, ...), mask: same shape but broadcastable on the reduced dim.
    """
    mask = mask.to(dtype=x.dtype)
    num = (x * mask).sum(dim=dim, keepdim=True)
    den = mask.sum(dim=dim, keepdim=True).clamp_min(eps)
    return num / den


def fourier_features(t: torch.Tensor, bands: int = 6) -> torch.Tensor:
    """Sin/Cos positional features for scalar t in [0,1].
    Returns shape t.shape + (4*bands,)."""
    # t: [..., 1]
    device = t.device
    freqs = 2.0 ** torch.arange(bands, device=device).float() * math.pi
    # [..., 1] * [bands] -> [..., bands]
    ang = t * freqs
    sin = torch.sin(ang)
    cos = torch.cos(ang)
    # also include doubled frequency for a bit more capacity
    ang2 = 2.0 * ang
    sin2 = torch.sin(ang2)
    cos2 = torch.cos(ang2)
    return torch.cat([sin, cos, sin2, cos2], dim=-1)


# ------------------------------------------------------------
# DeepSets-VAE (skeleton, non-autoregressive decoder)
#   Encoder: point -> contour -> glyph
#   Decoder: z + contour embedding + t_pos -> residual point
# ------------------------------------------------------------

class PointTokenMLP(nn.Module):
    def __init__(self, d_model: int = 64, pos_bands: int = 6):
        super().__init__()
        self.pos_bands = pos_bands
        # input: x,y + pos (4*bands) -> d_model
        d_in = 2 + 4 * pos_bands
        self.net = nn.Sequential(
            nn.Linear(d_in, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.SiLU(),
        )

    def forward(self, pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # pts: [B,K,L,2]; t: [B,K,L,1] in [0,1]
        pos = fourier_features(t, self.pos_bands)
        x = torch.cat([pts, pos], dim=-1)
        return self.net(x)


class ContourEncoder(nn.Module):
    def __init__(self, d_in: int = 64, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, token_mean: torch.Tensor) -> torch.Tensor:
        # token_mean: [B,K,1,d_in] (mean over L, kept dim)
        x = token_mean.squeeze(-2)
        return self.net(x)  # [B,K,d_out]


class GlyphEncoder(nn.Module):
    def __init__(self, d_in: int = 128, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, contour_mean: torch.Tensor) -> torch.Tensor:
        # contour_mean: [B,1,d_in] (mean over K, kept dim)
        return self.net(contour_mean.squeeze(-2))  # [B,d_out]


class DecoderMLP(nn.Module):
    def __init__(self, d_seed: int, pos_bands: int = 6):
        super().__init__()
        self.pos_bands = pos_bands
        d_in = d_seed + 4 * pos_bands
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, seed: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # seed: [B,K,1,d_seed] (broadcast along L), t: [B,K,L,1]
        pos = fourier_features(t, self.pos_bands)
        h = torch.cat([seed.expand(-1, -1, t.size(2), -1), pos], dim=-1)
        return self.net(h)  # [B,K,L,2]


class DeepSetVAE(nn.Module):
    def __init__(self,
                 K_max: int = 24,
                 L_max: int = 128,
                 d_point: int = 2,
                 d_token: int = 64,
                 d_contour: int = 128,
                 d_glyph: int = 256,
                 z_dim: int = 128,
                 pos_bands: int = 6,
                 beta_kl: float = 1.0):
        super().__init__()
        self.K_max = K_max
        self.L_max = L_max
        self.beta_kl = beta_kl

        # Encoders
        self.point_mlp = PointTokenMLP(d_model=d_token, pos_bands=pos_bands)
        self.contour_enc = ContourEncoder(d_in=d_token, d_out=d_contour)
        self.glyph_enc = GlyphEncoder(d_in=d_contour, d_out=d_glyph)

        # Latent heads
        self.to_mu = nn.Linear(d_glyph, z_dim)
        self.to_logvar = nn.Linear(d_glyph, z_dim)

        # Seed for each contour from z + contour embedding
        self.seed_mlp = nn.Sequential(
            nn.Linear(z_dim + d_contour, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.decoder = DecoderMLP(d_seed=128, pos_bands=pos_bands)

    @staticmethod
    def _build_t_grid(B: int, K: int, L: int, device: torch.device) -> torch.Tensor:
        # t in [0,1], shape [B,K,L,1]
        t = torch.linspace(0, 1, L, device=device).view(1, 1, L, 1).expand(B, K, L, 1)
        return t

    def encode(self, points: torch.Tensor, mask: torch.Tensor):
        # points: [B,K,L,2]; mask: [B,K,L]
        B, K, L, _ = points.shape
        t = self._build_t_grid(B, K, L, points.device)
        tokens = self.point_mlp(points, t)  # [B,K,L,d_token]
        # token masked mean over L
        token_mask = mask.unsqueeze(-1)  # [B,K,L,1]
        token_mean = masked_mean(tokens, token_mask, dim=2)  # [B,K,1,d_token]
        e_contour = self.contour_enc(token_mean)             # [B,K,d_contour]

        # contour masked mean over K (use mask any valid point)
        k_mask = (mask.any(dim=2)).unsqueeze(-1)  # [B,K,1]
        contour_mean = masked_mean(e_contour, k_mask, dim=1) # [B,1,d_contour]
        e_glyph = self.glyph_enc(contour_mean)               # [B,d_glyph]

        mu = self.to_mu(e_glyph)
        logvar = self.to_logvar(e_glyph)
        return mu, logvar, e_contour

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, e_contour: torch.Tensor, L: int, mask: torch.Tensor) -> torch.Tensor:
        # z: [B,zdim], e_contour: [B,K,d_contour]
        B, K, _ = e_contour.shape
        z_tiled = z.unsqueeze(1).expand(B, K, -1)
        seed = self.seed_mlp(torch.cat([z_tiled, e_contour], dim=-1))  # [B,K,128]
        seed = seed.unsqueeze(2)  # [B,K,1,128]
        t = self._build_t_grid(B, K, L, e_contour.device)              # [B,K,L,1]
        delta = self.decoder(seed, t)                                  # [B,K,L,2]
        return delta

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        points = batch["points"]  # [B,K,L,2]
        mask = batch["mask"]      # [B,K,L]
        B, K, L, _ = points.shape

        # --- Encode ---
        mu, logvar, e_contour = self.encode(points, mask)

        # Compute KL in float32 and clamp logvar to avoid exp overflow
        mu32 = mu.float()
        logvar32 = logvar.float().clamp_(-8.0, 8.0)  # clamp keeps std in [~0.018, ~2.98]

        # --- Sample --- (keep z in fp32, cast to points.dtype for decode)
        z = self.reparameterize(mu32, logvar32)          # fp32 sample
        z_cast = z.to(points.dtype)                      # cast for decoder

        # --- Decode ---
        delta = self.decode(z_cast, e_contour, L, mask)
        recon = points + delta

        # --- Losses ---
        m = mask.unsqueeze(-1).to(points.dtype)
        l1 = (m * (recon - points).abs()).sum() / m.sum().clamp_min(1.0)

        # KL per sample in fp32
        kl_per = -0.5 * (1.0 + logvar32 - mu32.pow(2) - logvar32.exp()).sum(dim=1)
        kl = kl_per.mean()
        total = l1 + self.beta_kl * kl

        return {
            "recon": recon,
            "delta": delta,
            "mu": mu32,
            "logvar": logvar32,
            "z": z,
            "loss_total": total,
            "loss_l1": l1,
            "loss_kl": kl,
        }

# ------------------------------------------------------------
# Quick shape/grad check when run directly
# ------------------------------------------------------------
if __name__ == "__main__":
    B, K, L = 2, 8, 64
    pts = torch.randn(B, K, L, 2)
    mask = torch.zeros(B, K, L, dtype=torch.bool)
    mask[:, :, :48] = True
    batch = {"points": pts, "mask": mask, "lengths": torch.full((B, K), 48)}

    model = DeepSetVAE(K_max=K, L_max=L)
    out = model(batch)
    print({k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in out.items() if k in ("recon", "delta", "mu", "logvar", "z")})
    out["loss_total"].backward()
    print("OK: backward passed")
