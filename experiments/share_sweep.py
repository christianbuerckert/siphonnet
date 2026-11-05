"""How much of each discharge should be passed on? Memory capacity vs. routed share.

With 55 % routed, every discharge triggers well under one follow-up discharge on average
(subcritical); activity is mostly input-driven. Raising the share pushes the network
towards critical branching — and, for a passive hydraulic network at share 1, towards
water that can only leave through bleed holes and overflows.

    python experiments/share_sweep.py [--n 100] [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from siphonnet import AbstractSiphonNet, PhysicalParams, PhysicalSiphonNet, pump_inflow, random_input_matrix, random_topology
from siphonnet.readout import memory_capacity

RESULTS = Path(__file__).resolve().parent.parent / "results"
SHARES = [0.0, 0.3, 0.55, 0.8, 0.9, 1.0]
TICKS = 2500


def run_job(job):
    model, share, n, seed = job
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, TICKS)
    ch = np.stack([u, 1.0 - u], axis=1)[None]
    W_in = random_input_matrix(2, n, rng)
    model_rng = np.random.default_rng(100 + seed)
    if model == "abstract":
        net = AbstractSiphonNet(random_topology(n, rng, share=share), model_rng)
        hold = 4
        rec = net.run(pump_inflow(ch, W_in, net.params.input_gain), hold=hold, record_every=hold)
    else:
        p = PhysicalParams()
        net = PhysicalSiphonNet(random_topology(n, rng, share=share, delay_steps=(1, 5)), model_rng, p)
        hold = int(round(2.0 / p.dt))
        rec = net.run(pump_inflow(ch, W_in, p.input_gain), hold=hold, record_every=hold)
    mc, _ = memory_capacity(rec.series("both"), u)
    rate = rec.counts[0, 100:].mean()  # discharges per container per tick
    saturated = float("nan")
    if model == "physical":  # no new pulse in the last 200 ticks although the container is well filled
        quiet = rec.counts[0, -200:].sum(axis=0) == 0
        high = rec.levels[0, -200:].mean(axis=0) > (net.h_on + net.h_off) / 2
        saturated = float(np.mean(quiet & high))
    return model, share, seed, mc, rate, saturated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    jobs = [(m, s, args.n, seed) for m in ("abstract", "physical") for s in SHARES for seed in range(args.seeds)]
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for model, share, seed, mc, rate, sat in pool.map(run_job, jobs):
            res.setdefault((model, share), []).append((mc, rate, sat))
    lines = [
        f"# Memory capacity vs. routed share (N={args.n}, features = levels + counts, {args.seeds} seeds)",
        "",
        "`rate` = discharges per container per tick. `never empties` = share of siphons without a new pulse "
        "in the last 200 ticks although they are well filled (saturated through-flow or stuck at overflow).",
        "",
        "| share | abstract MC | abstract rate | physical MC | physical rate | physical never empties |",
        "|---|---|---|---|---|---|",
    ]
    for s in SHARES:
        a, p = np.array(res[("abstract", s)]), np.array(res[("physical", s)])
        lines.append(
            f"| {s:.2f} | {a[:, 0].mean():.1f} ± {a[:, 0].std():.1f} | {a[:, 1].mean():.3f} "
            f"| {p[:, 0].mean():.1f} ± {p[:, 0].std():.1f} | {p[:, 1].mean():.3f} | {100 * p[:, 2].mean():.0f} % |"
        )
    text = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "share_sweep.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
