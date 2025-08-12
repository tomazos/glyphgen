from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast  # updated to new API
from tqdm import tqdm

# Make project root importable if run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import GlyphSetDataset, RandomGlyphSampler, glyph_collate  # type: ignore
from models.deepset_vae import DeepSetVAE  # type: ignore
from train.metrics import masked_l1, closure_error, chamfer_distance  # type: ignore


@dataclass
class TrainCfg:
    ftdb_path: str
    out_dir: str = "runs/vae_baseline"
    steps: int = 5000
    batch_size: int = 4
    K_max: int = 24
    L_max: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-2
    beta_kl: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    ckpt_every: int = 1000
    log_every: int = 50


def save_ckpt(model: nn.Module, opt: torch.optim.Optimizer, scaler: Optional[GradScaler], step: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'step': step,
    }
    torch.save(ckpt, out_dir / f"ckpt_{step:06d}.pt")


def train(cfg: TrainCfg):
    device = torch.device(cfg.device)

    ds = GlyphSetDataset(cfg.ftdb_path, K_max=cfg.K_max, L_max=cfg.L_max)
    sampler = RandomGlyphSampler(ds, epoch_size=cfg.steps * cfg.batch_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0,
                    collate_fn=glyph_collate, pin_memory=(device.type == 'cuda'))

    model = DeepSetVAE(K_max=cfg.K_max, L_max=cfg.L_max, beta_kl=cfg.beta_kl).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(device.type, enabled=cfg.amp and device.type == 'cuda')

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(cfg.steps), desc="train", ncols=100)
    it = iter(dl)

    for step in pbar:
        batch = next(it)
        points = batch['points'].to(device)
        mask = batch['mask'].to(device)

        opt.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=cfg.amp and device.type == 'cuda'):
            out = model({'points': points, 'mask': mask, 'lengths': batch['lengths'].to(device)})
            # Extra metrics
            l1 = masked_l1(out['recon'], points, mask)
            clos = closure_error(out['recon'], mask)
            cham = chamfer_distance(out['recon'], points, mask, mask)
            loss = out['loss_total'] + 0.0 * (l1 + clos + cham)  # they're already included/for logging
        scaler.scale(loss).backward()

        # Unscale and clip to prevent runaway grads
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(opt)
        scaler.update()

        if step % cfg.log_every == 0:
            pbar.set_postfix({
                'l_total': f"{loss.item():.4f}",
                'l1': f"{l1.item():.4f}",
                'kl': f"{out['loss_kl'].item():.4f}",
                'clos': f"{clos.item():.4f}",
                'cham': f"{cham.item():.4f}",
            })

        if (step + 1) % cfg.ckpt_every == 0:
            save_ckpt(model, opt, scaler, step + 1, out_dir)

    # final save
    save_ckpt(model, opt, scaler, cfg.steps, out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ftdb_path')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--L', type=int, default=128)
    ap.add_argument('--out', type=str, default='runs/vae_baseline')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--beta_kl', type=float, default=0.1)
    ap.add_argument('--no-amp', action='store_true')
    args = ap.parse_args()

    cfg = TrainCfg(
        ftdb_path=args.ftdb_path,
        out_dir=args.out,
        steps=args.steps,
        batch_size=args.bs,
        K_max=args.K,
        L_max=args.L,
        lr=args.lr,
        beta_kl=args.beta_kl,
        amp=not args.no_amp,
    )
    train(cfg)
