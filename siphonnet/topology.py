"""Who pours into whom.

Every siphon has a distributor under its outlet that splits the discharge between a few
target containers. Whatever is not routed drains away, so the network can never create
water: the routed share per siphon is at most 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Topology:
    """Directed routing graph.

    Siphon ``i`` sends ``fractions[i, e]`` of its discharge to ``targets[i, e]``, arriving
    ``delays[i, e]`` simulation steps later (pipe transport time).
    """

    targets: np.ndarray  # (n, k) int
    fractions: np.ndarray  # (n, k) float, row sums <= 1
    delays: np.ndarray  # (n, k) int, >= 1

    @property
    def n(self) -> int:
        return self.targets.shape[0]

    def without_recurrence(self) -> "Topology":
        """Same network with every distributor closed: all discharge drains away (ablation)."""
        return Topology(self.targets, np.zeros_like(self.fractions), self.delays)

    def routing_matrices(self) -> dict[int, sparse.csr_matrix]:
        """One sparse matrix per delay ``d`` with ``M_d[i, j]`` = share of i's outflow reaching j."""
        mats: dict[int, sparse.csr_matrix] = {}
        active = self.fractions > 0
        for d in np.unique(self.delays[active]):
            rows, edges = np.nonzero(active & (self.delays == d))
            cols = self.targets[rows, edges]
            vals = self.fractions[rows, edges]
            mats[int(d)] = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return mats


def gravity_topology(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    out_degree: int = 4,
    reach: int = 1,
    max_dx: int = 2,
    share: float = 0.55,
    delay_steps: tuple[int, int] = (1, 5),
) -> Topology:
    """Containers on a rows x cols grid, row 0 at the top; water only flows downhill.

    Siphon (r, c) pours into ``out_degree`` random containers of the next ``reach`` rows
    within ``max_dx`` columns. A passive gravity network has no loops: it is a cascade,
    and the bottom row drains away. Unit i sits at row ``i // cols``, column ``i % cols``.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be in [0, 1]")
    n = rows * cols
    targets = np.tile(np.arange(n)[:, None], (1, out_degree))  # unused slots point to self with zero share
    fractions = np.zeros((n, out_degree))
    lo, hi = delay_steps
    delays = rng.integers(lo, hi + 1, size=(n, out_degree))
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            cand = [rr * cols + cc for rr in range(r + 1, min(rows, r + 1 + reach))
                    for cc in range(max(0, c - max_dx), min(cols, c + max_dx + 1))]
            chosen = rng.choice(cand, size=min(out_degree, len(cand)), replace=False)
            targets[i, : chosen.size] = chosen
            fractions[i, : chosen.size] = share / chosen.size
    return Topology(targets, fractions, delays)


def random_topology(
    n: int,
    rng: np.random.Generator,
    out_degree: int = 7,
    share: float = 0.55,
    delay_steps: tuple[int, int] = (1, 1),
    equal_split: bool = True,
) -> Topology:
    """Random graph with ``out_degree`` distinct targets per siphon and no self-loops.

    ``share`` is the routed fraction of each discharge (the rest drains). With
    ``equal_split`` the share is divided evenly, otherwise by a random Dirichlet split.
    ``delay_steps`` is the inclusive range of transport delays in simulation steps.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be in [0, 1]: a passive network cannot create water")
    lo, hi = delay_steps
    if lo < 1 or hi < lo:
        raise ValueError("delays must be >= 1 step")
    k = min(out_degree, n - 1)
    targets = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        t = rng.choice(n - 1, size=k, replace=False)
        t[t >= i] += 1
        targets[i] = t
    if equal_split or k == 0:
        fractions = np.full((n, k), share / max(k, 1))
    else:
        fractions = rng.dirichlet(np.ones(k), size=n) * share
    delays = rng.integers(lo, hi + 1, size=(n, k))
    return Topology(targets, fractions, delays)
