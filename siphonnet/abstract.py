"""The abstract discrete-time model, as in the original prototype.

    h_i(t+1) = leak * h_i(t) + u_i(t) + sum_j M_ji q_j(t - d_ji)
    fires    if h_i >= theta_i:  q_i = h_i (the whole content), then h_i = 0

Mathematically this is a leaky integrate-and-fire network with reset to zero, graded
spikes (amplitude = content at firing), excitatory weights only and transport delays.
Nothing in it is specific to siphons; see ``physical.py`` for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._router import Router
from .recording import Recording
from .topology import Topology


@dataclass
class AbstractParams:
    leak: float = 0.96  # fraction of content kept per step (tau ~ 24.5 steps)
    threshold_range: tuple[float, float] = (0.85, 1.15)
    input_gain: float = 0.15  # inflow per step at full drive


class AbstractSiphonNet:
    def __init__(self, topology: Topology, rng: np.random.Generator, params: AbstractParams | None = None):
        self.topology = topology
        self.params = params or AbstractParams()
        self.theta = rng.uniform(*self.params.threshold_range, size=topology.n)

    def run(self, inflow: np.ndarray, hold: int, record_every: int, n_steps: int | None = None) -> Recording:
        """Simulate a batch.

        ``inflow`` has shape (B, S, n): input sample ``s`` is applied during steps
        ``[s * hold, (s + 1) * hold)``. States are summarised every ``record_every`` steps.
        """
        B, S, n = inflow.shape
        n_steps = n_steps or S * hold
        blocks = n_steps // record_every
        leak = self.params.leak
        router = Router(self.topology, B)
        h = np.zeros((B, n))
        counts = np.zeros((B, blocks, n), dtype=np.int32)
        levels = np.zeros((B, blocks, n))
        for t in range(n_steps):
            h = leak * h + inflow[:, min(t // hold, S - 1)] + router.arriving(t)
            fire = h >= self.theta
            router.send(t, np.where(fire, h, 0.0))
            h = np.where(fire, 0.0, h)
            b = t // record_every
            if b < blocks:
                counts[:, b] += fire
                levels[:, b] += h
        levels /= record_every
        return Recording(counts, levels)
