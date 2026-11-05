"""Two moons, done fairly: ablation (no recurrence) and baselines with equal feature counts.

The question is not "does SiphonNet beat logistic regression on raw inputs" (any nonlinear
feature map does), but "do the network dynamics add anything over the same siphons
without connections, or over random features without any dynamics?".

    python experiments/two_moons.py [--sizes 10 30 100 300 1000] [--seeds 5]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from siphonnet import (
    AbstractSiphonNet,
    ComplementEncoder,
    PhysicalParams,
    PhysicalSiphonNet,
    pump_inflow,
    random_input_matrix,
    random_topology,
)
from siphonnet.baselines import random_relu_features
from siphonnet.readout import classify

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_SAMPLES, NOISE = 1000, 0.2
ABSTRACT_STEPS, WINDOWS = 42, 3
PHYS_SECONDS = 60.0


def data(seed: int):
    X, y = make_moons(N_SAMPLES, noise=NOISE, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
    enc = ComplementEncoder().fit(X_tr)
    return enc.transform(X_tr), enc.transform(X_te), y_tr, y_te


def run_job(job: tuple[str, int, int]) -> tuple[str, int, int, float]:
    model, n, seed = job
    ch_tr, ch_te, y_tr, y_te = data(seed)
    ch = np.concatenate([ch_tr, ch_te])
    split = len(ch_tr)
    rng = np.random.default_rng(1000 + seed)
    W_in = random_input_matrix(ch.shape[1], n, rng)
    model_rng = np.random.default_rng(2000 + seed)  # same thresholds / tolerances with and without recurrence

    if model == "ELM (random ReLU, 3N features)":
        F = random_relu_features(ch[:, : ch.shape[1] // 2], WINDOWS * n, rng)
    elif model.startswith("abstract"):
        top = random_topology(n, rng)
        if "W=0" in model:
            top = top.without_recurrence()
        net = AbstractSiphonNet(top, model_rng)
        rec = net.run(pump_inflow(ch, W_in, net.params.input_gain), hold=ABSTRACT_STEPS,
                      record_every=ABSTRACT_STEPS // WINDOWS)
        F = rec.features("counts")
    elif model.startswith("physical"):
        p = PhysicalParams()
        top = random_topology(n, rng, delay_steps=(1, 5))
        if "W=0" in model:
            top = top.without_recurrence()
        net = PhysicalSiphonNet(top, model_rng, p)
        steps = int(PHYS_SECONDS / p.dt)
        rec = net.run(pump_inflow(ch, W_in, p.input_gain), hold=steps, record_every=steps // WINDOWS)
        F = rec.features("both" if "levels" in model else "counts")
    else:
        raise ValueError(model)
    return model, n, seed, classify(F[:split], y_tr, F[split:], y_te)


def fixed_baselines(seeds: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"logistic regression (raw x)": [], "polynomial deg 3 + logistic": [], "RBF-SVM": []}
    for seed in range(seeds):
        X, y = make_moons(N_SAMPLES, noise=NOISE, random_state=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)
        models = {
            "logistic regression (raw x)": make_pipeline(StandardScaler(), LogisticRegression()),
            "polynomial deg 3 + logistic": make_pipeline(PolynomialFeatures(3), StandardScaler(), LogisticRegression(max_iter=2000)),
            "RBF-SVM": make_pipeline(StandardScaler(), SVC()),
        }
        for name, m in models.items():
            out[name].append(m.fit(X_tr, y_tr).score(X_te, y_te))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 30, 100, 300, 1000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    models = [
        "abstract SiphonNet",
        "abstract SiphonNet, W=0",
        "physical SiphonNet",
        "physical SiphonNet, W=0",
        "physical SiphonNet + levels",
        "ELM (random ReLU, 3N features)",
    ]
    jobs = [(m, n, s) for n in args.sizes for m in models for s in range(args.seeds)]
    jobs.sort(key=lambda j: -j[1])  # big ones first
    acc: dict[tuple[str, int], list[float]] = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for model, n, seed, a in pool.map(run_job, jobs):
            acc.setdefault((model, n), []).append(a)

    fmt = lambda v: f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"
    lines = [
        f"# Two moons (n={N_SAMPLES}, noise={NOISE}, 50/50 split, {args.seeds} seeds)",
        "",
        "Test accuracy in % (mean ± std over seeds). N = number of siphons; "
        f"features = discharge counts in {WINDOWS} time windows (3N), '+ levels' adds mean fill levels (6N).",
        "",
        "| model | " + " | ".join(f"N={n}" for n in args.sizes) + " |",
        "|---|" + "---|" * len(args.sizes),
    ]
    for m in models:
        lines.append(f"| {m} | " + " | ".join(fmt(acc[(m, n)]) for n in args.sizes) + " |")
    lines += ["", "| baseline without siphons | accuracy |", "|---|---|"]
    for name, v in fixed_baselines(args.seeds).items():
        lines.append(f"| {name} | {fmt(v)} |")
    text = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "two_moons.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
