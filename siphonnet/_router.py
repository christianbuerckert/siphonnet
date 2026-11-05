"""Delay line for routed discharges (shared by both simulators)."""

from __future__ import annotations

import numpy as np

from .topology import Topology


class Router:
    """Ring buffer of past outflows; returns what arrives at each container now."""

    def __init__(self, topology: Topology, batch: int):
        self.mats = topology.routing_matrices()
        self.length = (max(self.mats) if self.mats else 0) + 1
        self.hist = np.zeros((self.length, batch, topology.n))

    def arriving(self, t: int) -> np.ndarray | float:
        rec: np.ndarray | float = 0.0
        for d, M in self.mats.items():
            rec = rec + self.hist[(t - d) % self.length] @ M
        return rec

    def send(self, t: int, outflow: np.ndarray) -> None:
        self.hist[t % self.length] = outflow
