"""What a readout can see: discharge counts and water levels per time block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Recording:
    counts: np.ndarray  # (B, blocks, n) discharges started within each block
    levels: np.ndarray  # (B, blocks, n) mean state (fill level) within each block
    primed: np.ndarray | None = None  # (B, blocks, n) fraction of the block the siphon was flowing (physical model)

    def features(self, kind: str = "counts") -> np.ndarray:
        """Flattened per-trial feature vectors, shape (B, blocks * n * parts)."""
        parts = {"counts": [self.counts], "levels": [self.levels], "both": [self.counts, self.levels]}[kind]
        return np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=1).astype(float)

    def series(self, kind: str = "levels") -> np.ndarray:
        """Per-block states of a single long run, shape (blocks, n * parts) — for temporal tasks."""
        parts = {"counts": [self.counts], "levels": [self.levels], "both": [self.counts, self.levels]}[kind]
        return np.concatenate([p[0] for p in parts], axis=1).astype(float)
