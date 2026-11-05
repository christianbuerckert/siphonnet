"""Simulate everything the demo video shows and store it in video/build/data.npz.

Nothing in the video is animated by hand except the look of water streams: levels,
pulses and the recognised letters all come from here.

    python video/prepare.py [--word SIPHON]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from siphonnet import PhysicalParams, PhysicalSiphonNet, Topology
from siphonnet.glyphs import LETTERS, noisy
from siphonnet.letters import build, dataset

BUILD = Path(__file__).resolve().parent / "build"
N, ROWS, MACHINE_SEED = 100, 5, 7
SINGLE_INFLOW, SINGLE_SECONDS, SINGLE_DT = 8.0, 20.0, 0.05


def single_siphon() -> dict:
    p = PhysicalParams(dt=SINGLE_DT, area_spread=0.0, level_spread=0.0)
    top = Topology(np.zeros((1, 0), int), np.zeros((1, 0)), np.ones((1, 0), int))
    net = PhysicalSiphonNet(top, np.random.default_rng(0), p)
    steps = int(SINGLE_SECONDS / p.dt)
    rec = net.run(np.full((1, 1, 1), SINGLE_INFLOW), hold=steps, record_every=1)
    return {
        "single_level": rec.levels[0, :, 0],
        "single_flowing": rec.primed[0, :, 0],
        "single_starts": rec.counts[0, :, 0],
        "single_h_on": p.h_on,
        "single_h_off": p.h_off,
        "single_h_max": p.h_max,
        "single_dt": p.dt,
    }


def network(word: str) -> tuple[dict, dict]:
    rng = np.random.default_rng(1)
    machine = build(N, seed=MACHINE_SEED, rows=ROWS)
    X_tr, y_tr = dataset(40, rng)
    X_te, y_te = dataset(20, rng)
    machine.fit(X_tr, y_tr)
    accuracy = float(np.mean(machine.predict(X_te) == y_te))
    target = np.array([LETTERS.index(c) for c in word])
    # first demo seed whose six noisy samples are all recognised (disclosed in the README)
    for demo_seed in range(100):
        r = np.random.default_rng(10_000 + demo_seed)
        pumps = np.array([noisy(c, r) for c in word])
        if np.all(machine.predict(pumps) == target):
            break
    rec = machine.run(pumps, record_every=1)
    net = machine.net
    # what each container contributes to each letter's score: coef * standardised feature, summed per container
    scaler, ridge = machine.readout[0], machine.readout[-1]
    z = scaler.transform(machine.features(pumps)).reshape(len(word), 2, -1, N)  # (B, counts/levels, windows, n)
    coef = ridge.coef_.reshape(len(LETTERS), 2, -1, N)
    contributions = np.einsum("jpwn,bpwn->bjn", coef, z)
    data = {
        "word_pumps": pumps,
        "levels": rec.levels,
        "flowing": rec.primed,
        "starts": rec.counts,
        "scores": machine.scores(pumps),
        "inflow": pumps @ machine.W_in * net.params.input_gain,
        "contributions": contributions,
        "w_in": machine.W_in,
        "rows": ROWS,
        "cols": N // ROWS,
        "targets": net.topology.targets,
        "fractions": net.topology.fractions,
        "h_on": net.h_on,
        "h_off": net.h_off,
        "h_max": net.params.h_max,
        "net_dt": net.params.dt,
    }
    meta = {"word": word, "accuracy": accuracy, "demo_seed": demo_seed, "n": N, "machine_seed": MACHINE_SEED}
    return data, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="SIPHON")
    args = ap.parse_args()
    word = "".join(c for c in args.word.upper() if c in LETTERS)
    data, meta = network(word)
    data.update(single_siphon())
    BUILD.mkdir(exist_ok=True)
    np.savez_compressed(BUILD / "data.npz", **data)
    (BUILD / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
