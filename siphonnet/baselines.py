"""Digital baselines a siphon reservoir has to beat (or at least match)."""

from __future__ import annotations

import numpy as np


def random_relu_features(
    X: np.ndarray, n_features: int, rng: np.random.Generator, scale: float = 3.0
) -> np.ndarray:
    """Extreme learning machine: a fixed random projection and a ReLU, no dynamics.

    If a reservoir does no better than this, its dynamics add nothing for the task.
    ``X`` should be scaled to roughly [0, 1].
    """
    d = X.shape[1]
    W = rng.normal(0.0, scale, (d, n_features))
    b = rng.uniform(-scale, scale, n_features) * np.sqrt(d) / 2
    return np.maximum(0.0, X @ W + b)


class EchoStateNetwork:
    """Plain tanh echo state network (Jaeger 2001) with sparse random recurrence."""

    def __init__(
        self,
        n: int,
        n_inputs: int,
        rng: np.random.Generator,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        density: float = 0.1,
        leak: float = 1.0,
    ):
        W = rng.normal(0.0, 1.0, (n, n)) * (rng.random((n, n)) < density)
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius) if radius > 0 else W
        self.W_in = rng.uniform(-input_scale, input_scale, (n_inputs, n))
        self.bias = rng.uniform(-0.1, 0.1, n)
        self.leak = leak

    def run(self, u: np.ndarray) -> np.ndarray:
        """``u`` (T, n_inputs) -> states (T, n)."""
        x = np.zeros(self.W.shape[0])
        out = np.empty((u.shape[0], x.size))
        for t in range(u.shape[0]):
            x = (1 - self.leak) * x + self.leak * np.tanh(u[t] @ self.W_in + x @ self.W + self.bias)
            out[t] = x
        return out
