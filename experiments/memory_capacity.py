"""Linear short-term memory capacity: what a reservoir is actually for.

Input: i.i.d. uniform values u(t) in [0, 1], one per clock tick, fed as pump flows
[u, 1 - u]. A linear readout of the state at tick t must recall u(t - k), k = 1..40.
Memory capacity MC = sum of squared test correlations (Jaeger 2002); MC <= #features.

    python experiments/memory_capacity.py [--n 100] [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from siphonnet import (
    AbstractSiphonNet,
    PhysicalParams,
    PhysicalSiphonNet,
    pump_inflow,
    random_input_matrix,
    random_topology,
)
from siphonnet.baselines import EchoStateNetwork
from siphonnet.readout import memory_capacity

RESULTS = Path(__file__).resolve().parent.parent / "results"
TICKS, MAX_DELAY = 2500, 40
ABSTRACT_HOLD = 4  # steps per tick
PHYS_TICK = 2.0  # seconds per tick

MODELS = [
    ("ESN (tanh)", "states"),
    ("abstract SiphonNet", "levels"),
    ("abstract SiphonNet", "both"),
    ("abstract SiphonNet, W=0", "levels"),
    ("abstract SiphonNet, W=0", "both"),
    ("physical SiphonNet", "levels"),
    ("physical SiphonNet", "both"),
    ("physical SiphonNet, W=0", "levels"),
    ("physical SiphonNet, W=0", "both"),
    ("physical, sealed (no bleed)", "both"),
    ("physical, displacers 0-0.7", "both"),
    ("physical, tick 0.5 s", "both"),
    ("physical, tick 8 s", "both"),
]


def run_job(job):
    (model, kind), n, seed = job
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, TICKS)
    ch = np.stack([u, 1.0 - u], axis=1)[None]  # (1, T, 2)
    W_in = random_input_matrix(2, n, rng)
    model_rng = np.random.default_rng(100 + seed)

    if model.startswith("ESN"):
        states = EchoStateNetwork(n, 1, rng).run((u - 0.5)[:, None])
    elif model.startswith("abstract"):
        top = random_topology(n, rng)
        if "W=0" in model:
            top = top.without_recurrence()
        net = AbstractSiphonNet(top, model_rng)
        rec = net.run(pump_inflow(ch, W_in, net.params.input_gain), hold=ABSTRACT_HOLD, record_every=ABSTRACT_HOLD)
        states = rec.series(kind)
    else:
        p = PhysicalParams()
        tick = PHYS_TICK
        displacer = None
        if "sealed" in model:
            p = replace(p, bleed_diameter=0.0)
        if "tick" in model:
            tick = float(model.split("tick ")[1].split(" s")[0])
        if "displacers" in model:
            displacer = model_rng.uniform(0.0, 0.7, n)
        top = random_topology(n, rng, delay_steps=(1, 5))
        if "W=0" in model:
            top = top.without_recurrence()
        net = PhysicalSiphonNet(top, model_rng, p, displacer=displacer)
        hold = int(round(tick / p.dt))
        rec = net.run(pump_inflow(ch, W_in, p.input_gain), hold=hold, record_every=hold)
        states = rec.series(kind)
    mc, r2 = memory_capacity(states, u, max_delay=MAX_DELAY)
    return (model, kind), seed, mc, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    jobs = [(m, args.n, s) for m in MODELS for s in range(args.seeds)]
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for key, seed, mc, r2 in pool.map(run_job, jobs):
            res.setdefault(key, []).append((mc, r2))

    lines = [
        f"# Linear memory capacity (N={args.n}, {TICKS} ticks, delays 1..{MAX_DELAY}, {args.seeds} seeds)",
        "",
        "Features per tick: `levels` = mean fill level of every container (N), `both` = levels + discharge "
        "counts (2N), `states` = ESN activations (N). Abstract model: "
        f"{ABSTRACT_HOLD} steps per tick. Physical model: {PHYS_TICK:g} s per tick unless stated.",
        "",
        "| model | features | MC | delay at which r² < 0.5 |",
        "|---|---|---|---|",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in MODELS:
        mcs = np.array([m for m, _ in res[key]])
        r2 = np.mean([r for _, r in res[key]], axis=0)
        below = np.nonzero(r2 < 0.5)[0]
        horizon = f"{below[0] + 1}" if below.size else f"> {MAX_DELAY}"
        lines.append(f"| {key[0]} | {key[1]} | {mcs.mean():.1f} ± {mcs.std():.1f} | {horizon} |")
        if key[1] in ("states", "both") and "tick" not in key[0]:
            ax.plot(np.arange(1, MAX_DELAY + 1), r2, label=f"{key[0]} ({key[1]})")
    ax.set_xlabel("delay k (ticks)")
    ax.set_ylabel("recall r² of u(t − k)")
    ax.set_title(f"Memory curves, N = {args.n}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "memory_capacity.png", dpi=130)
    text = "\n".join(lines) + "\n"
    (RESULTS / "memory_capacity.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
