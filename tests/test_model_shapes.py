import os
import sys
from pathlib import Path

import torch

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import GlyphSetDataset, glyph_collate, RandomGlyphSampler  # type: ignore
from models.deepset_vae import DeepSetVAE  # type: ignore
from torch.utils.data import DataLoader


def test_model_forward_backward():
    ftdb = os.environ.get("FTDB_PATH")
    if not ftdb or not Path(ftdb).exists():
        import pytest
        pytest.skip("Set FTDB_PATH env var to run model shape test")

    # Tiny dataset & loader
    K, L = 8, 64
    ds = GlyphSetDataset(ftdb, K_max=K, L_max=L)
    dl = DataLoader(ds, batch_size=2, sampler=RandomGlyphSampler(ds, epoch_size=2),
                    num_workers=0, collate_fn=glyph_collate)

    batch = next(iter(dl))

    model = DeepSetVAE(K_max=K, L_max=L)
    out = model(batch)

    # Check required keys/ shapes
    for key in ("recon", "delta", "mu", "logvar", "z", "loss_total", "loss_l1", "loss_kl"):
        assert key in out, f"missing key: {key}"
    B = batch["points"].shape[0]
    assert out["recon"].shape == batch["points"].shape
    assert out["delta"].shape == batch["points"].shape
    assert out["mu"].shape == (B, 128)
    assert out["logvar"].shape == (B, 128)

    # Backward pass shouldn't error
    out["loss_total"].backward()
    # Check some parameter got a grad
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
