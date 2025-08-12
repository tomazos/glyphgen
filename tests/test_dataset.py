import os
import sys
from pathlib import Path

import torch

# Make project root importable when running pytest from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import GlyphSetDataset, glyph_collate, RandomGlyphSampler  # type: ignore


def test_dataset_smoke(tmp_path: Path):
    # This is a smoke test that only checks wiring/shapes. It requires a real FTDB path via env var.
    ftdb = os.environ.get("FTDB_PATH")
    if not ftdb or not Path(ftdb).exists():
        import pytest
        pytest.skip("Set FTDB_PATH env var to run dataset smoke test")

    ds = GlyphSetDataset(ftdb, K_max=8, L_max=64)
    # Single item
    item = ds[0]
    assert item.points.shape == (8, 64, 2)
    assert item.lengths.shape == (8,)
    assert item.mask.shape == (8, 64)
    assert item.num_contours.shape == ()

    # Collate a small batch
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=2, sampler=RandomGlyphSampler(ds, epoch_size=2),
                    num_workers=0, collate_fn=glyph_collate)
    batch = next(iter(dl))
    assert batch["points"].shape == (2, 8, 64, 2)
    assert batch["lengths"].shape == (2, 8)
    assert batch["mask"].shape == (2, 8, 64)
    assert batch["num_contours"].shape == (2,)
    assert batch["glyph_index"].shape == (2,)

    # dtype sanity
    assert batch["points"].dtype == torch.float32
    assert batch["lengths"].dtype == torch.long
    assert batch["mask"].dtype == torch.bool
