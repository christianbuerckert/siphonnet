"""Letter recognition: 5x7 pixel glyphs as pumps, a siphon network, a trained readout.

This is the task behind the demo video. Each lit pixel drives one pump (with ±20 %
pump tolerance, and three pixels flipped at random); the network runs for a fixed
time; a ridge classifier reads pulse counts and fill levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .glyphs import LETTERS, noisy
from .inputs import pump_inflow, random_input_matrix
from .physical import PhysicalParams, PhysicalSiphonNet
from .readout import ALPHAS
from .recording import Recording
from .topology import gravity_topology, random_topology

TRIAL_SECONDS = 30.0
WINDOWS = 3


def dataset(per_letter: int, rng: np.random.Generator, flips: int = 3) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([noisy(c, rng, flips) for c in LETTERS for _ in range(per_letter)])
    y = np.repeat(np.arange(len(LETTERS)), per_letter)
    return X, y


@dataclass
class LetterMachine:
    net: PhysicalSiphonNet
    W_in: np.ndarray
    readout: Pipeline | None = None

    @property
    def steps(self) -> int:
        return int(TRIAL_SECONDS / self.net.params.dt)

    def run(self, pumps: np.ndarray, record_every: int | None = None) -> Recording:
        """``pumps`` (B, 35) -> recording; per window by default, per step for animation."""
        p = self.net.params
        return self.net.run(pump_inflow(pumps, self.W_in, p.input_gain), hold=self.steps,
                            record_every=record_every or self.steps // WINDOWS)

    def features(self, pumps: np.ndarray) -> np.ndarray:
        return self.run(pumps).features("both")

    def fit(self, pumps: np.ndarray, y: np.ndarray) -> "LetterMachine":
        self.readout = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS)).fit(self.features(pumps), y)
        return self

    def scores(self, pumps: np.ndarray) -> np.ndarray:
        return self.readout.decision_function(self.features(pumps))

    def predict(self, pumps: np.ndarray) -> np.ndarray:
        return self.readout.predict(self.features(pumps))


def build(n: int = 100, seed: int = 7, recurrent: bool = True, params: PhysicalParams | None = None,
          layout: str = "gravity", rows: int | None = None) -> LetterMachine:
    """``layout="gravity"``: grid of ``rows`` (default: square), water only flows to lower rows
    (physically buildable). ``layout="random"``: any container may feed any other (needs pumps
    between them). Pumps feed containers in every row."""
    rng = np.random.default_rng(seed)
    if layout == "gravity":
        rows = rows or int(round(np.sqrt(n)))
        if n % rows:
            raise ValueError("n must be a multiple of rows")
        top = gravity_topology(rows, n // rows, rng)
    else:
        top = random_topology(n, rng, delay_steps=(1, 5))
    if not recurrent:
        top = top.without_recurrence()
    W_in = random_input_matrix(35, n, rng, density=0.15)
    net = PhysicalSiphonNet(top, np.random.default_rng(seed + 1), params or PhysicalParams())
    return LetterMachine(net, W_in)
