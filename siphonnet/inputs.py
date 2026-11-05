"""How data becomes water.

Flows cannot be negative, so every real-valued feature x is min-max scaled to [0, 1] and
drives two pumps, x and 1 - x (complement coding). Each container receives a random,
non-negative mix of pump channels. This mapping is linear: any nonlinearity in the
features has to come from the siphons, not from the encoding.
"""

from __future__ import annotations

import numpy as np


class ComplementEncoder:
    """Min-max scaling fitted on training data, then channels [x, 1 - x]."""

    def fit(self, X: np.ndarray) -> "ComplementEncoder":
        X = np.asarray(X, dtype=float)
        self.lo_ = X.min(axis=0)
        span = X.max(axis=0) - self.lo_
        self.span_ = np.where(span > 0, span, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.clip((np.asarray(X, dtype=float) - self.lo_) / self.span_, 0.0, 1.0)
        return np.concatenate([z, 1.0 - z], axis=-1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def random_input_matrix(
    n_channels: int, n: int, rng: np.random.Generator, density: float = 0.5
) -> np.ndarray:
    """Non-negative pump-to-container matrix of shape (channels, n).

    Each container gets a random mix of channels (columns sum to a random gain in
    [0.5, 1.5]) so containers differ in how strongly they are fed.
    """
    W = rng.uniform(0.0, 1.0, (n_channels, n)) * (rng.random((n_channels, n)) < density)
    empty = np.nonzero(W.sum(axis=0) == 0)[0]
    W[rng.integers(0, n_channels, empty.size), empty] = rng.uniform(0.5, 1.0, empty.size)
    W /= W.sum(axis=0, keepdims=True)
    return W * rng.uniform(0.5, 1.5, n)


def pump_inflow(channels: np.ndarray, W_in: np.ndarray, gain: float) -> np.ndarray:
    """Inflow per container: ``channels`` (B, S, C) -> (B, S, n), scaled by ``gain``."""
    channels = np.asarray(channels, dtype=float)
    if channels.ndim == 2:  # static task: one input sample per trial
        channels = channels[:, None, :]
    return gain * (channels @ W_in)
