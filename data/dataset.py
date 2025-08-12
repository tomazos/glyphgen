from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# Import the FTDB reader from the project layout (glyphgen/data/ftdb_reader.py)
try:
    from .ftdb_reader import FTDB
except Exception:
    # Fallback if dataset.py is run standalone
    from data.ftdb_reader import FTDB  # type: ignore


# ------------------------------
# Utilities
# ------------------------------

def _contour_area(points: np.ndarray) -> float:
    """Signed area via the shoelace formula. points: [N,2] closed implicitly."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    s = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return 0.5 * s


def _normalize_glyph(contours: List[np.ndarray]) -> List[np.ndarray]:
    """Center by centroid and scale to unit box (max|x|,|y| = 1)."""
    if not contours:
        return contours
    all_pts = np.concatenate(contours, axis=0)
    centroid = all_pts.mean(axis=0)
    contours = [c - centroid for c in contours]
    all_pts2 = np.concatenate(contours, axis=0)
    denom = float(np.max(np.abs(all_pts2)))
    if denom > 0:
        s = 1.0 / denom
        contours = [c * s for c in contours]
    # Ensure consistent winding (positive area) by flipping if total area < 0
    total_area = sum(_contour_area(c) for c in contours)
    if total_area < 0:
        contours = [c[::-1].copy() for c in contours]
    return contours


@dataclass
class GlyphSample:
    # Padded tensors with masks (K = max_contours, L = max_points)
    points: torch.Tensor      # [K, L, 2] (float32), padded with 0s
    lengths: torch.Tensor     # [K] (int64) actual point counts per contour (even)
    mask: torch.Tensor        # [K, L] bool; True where valid
    num_contours: torch.Tensor # [] (int64)
    glyph_index: int


class GlyphSetDataset(Dataset):
    """Streams random glyphs from an FTDB file as sets of closed contours.

    Returns padded per-glyph tensors suitable for batching.
    """

    def __init__(
        self,
        ftdb_path: str,
        K_max: int = 24,
        L_max: int = 128,
        drop_empty: bool = True,
        normalize: bool = True,
        rng: Optional[random.Random] = None,
        glyph_indices: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        self.db = FTDB(ftdb_path)
        self.K_max = int(K_max)
        self.L_max = int(L_max)
        self.drop_empty = bool(drop_empty)
        self.normalize = bool(normalize)
        self.rng = rng or random.Random()
        # Candidate glyph indices to sample from; default = all glyphs
        if glyph_indices is None:
            self._indices = np.arange(self.db.glyph_count, dtype=np.int64)
        else:
            self._indices = np.fromiter(glyph_indices, dtype=np.int64)
        if len(self._indices) == 0:
            raise ValueError("No glyph indices provided")

    def __len__(self) -> int:
        # Treat as virtually infinite; iterate as many times as needed.
        return len(self._indices)

    def _read_glyph_contours(self, gi: int) -> List[np.ndarray]:
        g = self.db.glyph(int(gi))
        contours: List[np.ndarray] = []
        for cidx in range(g.first_contour, g.first_contour + g.num_contours):
            ci = self.db.contour(cidx)
            if ci.num_points <= 0 or (ci.num_points % 2) != 0:
                continue
            pts = np.empty((ci.num_points, 2), dtype=np.float32)
            base = self.db._off_pnts_data + ci.first_point * 8  # 2*f32 = 8 bytes per point
            # read directly from the mmap for speed
            for j in range(ci.num_points):
                # Unpack float32 little-endian; manual to avoid Python loop overhead? Acceptable for now.
                x, y = self.db.point(ci.first_point + j)
                pts[j, 0] = x
                pts[j, 1] = y
            contours.append(pts)
        return contours

    def _choose_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        if not contours:
            return []
        # Truncate very long contours to L_max (even length)
        trimmed: List[np.ndarray] = []
        for c in contours:
            L = c.shape[0]
            if L > self.L_max:
                # keep even truncation length
                t = self.L_max if (self.L_max % 2 == 0) else (self.L_max - 1)
                trimmed.append(c[:t])
            else:
                trimmed.append(c)
        # If too many contours, pick the K_max with largest |area|
        if len(trimmed) > self.K_max:
            areas = [abs(_contour_area(c)) for c in trimmed]
            idx = np.argsort(areas)[-self.K_max:]
            trimmed = [trimmed[i] for i in idx]
        return trimmed

    def _pack(self, contours: List[np.ndarray], glyph_index: int) -> GlyphSample:
        K = min(len(contours), self.K_max)
        points = torch.zeros((self.K_max, self.L_max, 2), dtype=torch.float32)
        lengths = torch.zeros((self.K_max,), dtype=torch.long)
        mask = torch.zeros((self.K_max, self.L_max), dtype=torch.bool)
        for k in range(K):
            c = contours[k]
            L = c.shape[0]
            points[k, :L, :] = torch.from_numpy(c)
            lengths[k] = L
            mask[k, :L] = True
        return GlyphSample(points=points, lengths=lengths, mask=mask,
                           num_contours=torch.tensor(K, dtype=torch.long),
                           glyph_index=int(glyph_index))

    def __getitem__(self, idx: int) -> GlyphSample:
        # Map incoming index to a random glyph to encourage randomization even with a sequential sampler
        gi = int(self._indices[self.rng.randrange(0, len(self._indices))])
        contours = self._read_glyph_contours(gi)
        if self.drop_empty:
            contours = [c for c in contours if c.shape[0] >= 4]
        if self.normalize:
            contours = _normalize_glyph(contours)
        contours = self._choose_contours(contours)
        # Resample if empty after filtering
        tries = 0
        while not contours and tries < 10:
            gi = int(self._indices[self.rng.randrange(0, len(self._indices))])
            contours = self._read_glyph_contours(gi)
            if self.drop_empty:
                contours = [c for c in contours if c.shape[0] >= 4]
            if self.normalize:
                contours = _normalize_glyph(contours)
            contours = self._choose_contours(contours)
            tries += 1
        return self._pack(contours, gi)


class RandomGlyphSampler(Sampler[int]):
    """Infinite/random sampler over glyph indices (by dataset row)."""

    def __init__(self, data_source: GlyphSetDataset, epoch_size: int = 100000):
        self.data_source = data_source
        self.epoch_size = int(epoch_size)
        self.rng = random.Random()

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.epoch_size):
            yield self.rng.randrange(0, len(self.data_source))

    def __len__(self) -> int:
        return self.epoch_size


def glyph_collate(batch: List[GlyphSample]) -> Dict[str, torch.Tensor]:
    # Stack fields
    points = torch.stack([b.points for b in batch], dim=0)       # [B,K,L,2]
    lengths = torch.stack([b.lengths for b in batch], dim=0)     # [B,K]
    mask = torch.stack([b.mask for b in batch], dim=0)           # [B,K,L]
    K = torch.stack([b.num_contours for b in batch], dim=0)      # [B]
    glyph_idx = torch.tensor([b.glyph_index for b in batch], dtype=torch.long)
    return {
        "points": points,
        "lengths": lengths,
        "mask": mask,
        "num_contours": K,
        "glyph_index": glyph_idx,
    }


# ------------------------------
# Quick self-test (optional)
# ------------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    ds = GlyphSetDataset(args.path, K_max=args.K, L_max=args.L)
    dl = DataLoader(ds, batch_size=args.bs, sampler=RandomGlyphSampler(ds, epoch_size=32),
                    num_workers=0, collate_fn=glyph_collate)
    for i, batch in enumerate(dl):
        print({k: tuple(v.shape) for k, v in batch.items()})
        if i >= 2:
            break
