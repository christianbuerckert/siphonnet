"""A bell-siphon network with the physics the abstract model leaves out.

Each container (cross-section A, effective A_eff = A * (1 - displacer)) holds a bell
siphon. State: fill level h [cm] and whether the siphon is primed.

    A_eff dh/dt = Q_pump + Q_routed - primed * Q_siphon(h) - Q_bleed(h)

    Q_siphon(h) = Cd * a_tube  * sqrt(2 g (h + drop))   Torricelli, head down to the outlet end
    Q_bleed(h)  = Cd * a_bleed * sqrt(2 g h)            optional small hole in the floor

    primes  when h >= h_on   (water spills over the standpipe and fills the bell)
    breaks  when h <= h_off  (air enters under the bell rim)

What this adds over integrate-and-fire:

* Hysteresis: the siphon fires at h_on and drains only down to h_off, not to zero.
* Finite discharge: a pulse takes seconds and its flow depends on the level.
* Inflow during discharge is not lost; it lengthens the pulse.
* Saturation: if inflow exceeds the siphon's flow at h_off, it never breaks and turns
  into a continuous through-flow.
* Overflow at h_max (excess goes to waste).
* No leak unless a bleed hole is drilled: a closed container is a perfect integrator.
* Displacer piston in the working zone (between h_off and h_on): it shrinks A_eff, so
  the container fills faster *and* releases smaller pulses. Mean throughput is
  unchanged (mass conservation); only the granularity changes. A piston below h_off
  would only fill dead volume and change nothing.

Units: cm, s, mL (= cm^3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._router import Router
from .recording import Recording
from .topology import Topology

G = 981.0  # cm / s^2


@dataclass
class PhysicalParams:
    dt: float = 0.1  # s
    area: float = 10.0  # cm^2, container cross-section
    area_spread: float = 0.05  # relative std, fabrication tolerance
    h_on: float = 8.0  # cm, siphon primes
    h_off: float = 2.0  # cm, siphon breaks
    level_spread: float = 0.05  # relative std of h_on / h_off
    h_max: float = 10.0  # cm, overflow edge
    drop: float = 10.0  # cm, outlet end below the container floor
    tube_diameter: float = 0.6  # cm
    bleed_diameter: float = 0.06  # cm, 0 = sealed container (no leak)
    discharge_coeff: float = 0.6
    displacer: float = 0.0  # fraction of the cross-section taken by a displacer piston
    input_gain: float = 6.0  # mL/s pump flow at full drive

    def orifice(self, diameter: float) -> float:
        """Cd * a * sqrt(2 g): flow in mL/s per sqrt(cm) of head."""
        return self.discharge_coeff * np.pi * diameter**2 / 4 * np.sqrt(2 * G)


class PhysicalSiphonNet:
    def __init__(
        self,
        topology: Topology,
        rng: np.random.Generator,
        params: PhysicalParams | None = None,
        displacer: np.ndarray | float | None = None,
    ):
        p = self.params = params or PhysicalParams()
        n = topology.n
        self.topology = topology
        jitter = lambda spread: np.clip(1 + spread * rng.standard_normal(n), 0.5, 1.5)
        self.area = p.area * jitter(p.area_spread)
        self.h_on = p.h_on * jitter(p.level_spread)
        self.h_off = np.minimum(p.h_off * jitter(p.level_spread), 0.9 * self.h_on)
        self.displacer = np.broadcast_to(np.asarray(p.displacer if displacer is None else displacer, float), (n,))
        if np.any((self.displacer < 0) | (self.displacer >= 1)):
            raise ValueError("displacer fraction must be in [0, 1)")
        self.c_siphon = p.orifice(p.tube_diameter)
        self.c_bleed = p.orifice(p.bleed_diameter)

    @property
    def area_eff(self) -> np.ndarray:
        return self.area * (1.0 - self.displacer)

    @property
    def pulse_volume(self) -> np.ndarray:
        """Water released per pulse when nothing flows in meanwhile [mL]."""
        return self.area_eff * (self.h_on - self.h_off)

    def run(self, inflow: np.ndarray, hold: int, record_every: int, n_steps: int | None = None) -> Recording:
        """Simulate a batch; ``inflow`` (B, S, n) in mL/s, sample ``s`` held for ``hold`` steps.

        All containers start at their break level h_off with the siphon unprimed, i.e.
        just after a pulse.
        """
        p = self.params
        B, S, n = inflow.shape
        n_steps = n_steps or S * hold
        blocks = n_steps // record_every
        dt, A = p.dt, self.area_eff
        router = Router(self.topology, B)
        h = np.broadcast_to(self.h_off, (B, n)).copy()
        primed = np.zeros((B, n), dtype=bool)
        counts = np.zeros((B, blocks, n), dtype=np.int32)
        levels = np.zeros((B, blocks, n))
        flowing = np.zeros((B, blocks, n))
        for t in range(n_steps):
            q_in = inflow[:, min(t // hold, S - 1)] + router.arriving(t)
            q_siphon = np.where(primed, self.c_siphon * np.sqrt(h + p.drop), 0.0)
            q_bleed = self.c_bleed * np.sqrt(h)
            h_new = h + dt * (q_in - q_siphon - q_bleed) / A
            # the siphon cannot pull the level below its break point within one step
            under = primed & (h_new < self.h_off)
            q_siphon = np.where(under, np.maximum(0.0, q_siphon - (self.h_off - h_new) * A / dt), q_siphon)
            h = np.clip(np.where(under, self.h_off, h_new), 0.0, p.h_max)
            starts = ~primed & (h >= self.h_on)
            primed = (primed | starts) & ~(h <= self.h_off)
            router.send(t, q_siphon)
            b = t // record_every
            if b < blocks:
                counts[:, b] += starts
                levels[:, b] += h
                flowing[:, b] += q_siphon > 0
        levels /= record_every
        flowing /= record_every
        return Recording(counts, levels, flowing)
