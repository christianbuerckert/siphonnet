"""Letter recognition (A-Z, 5x7 pixels, 3 flipped pixels, ±20 % pump tolerance).

Compares the physical siphon network with and without connections against a linear
classifier on the raw pixels.

    python experiments/letters.py [--sizes 36 64 100] [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from siphonnet.letters import build, dataset
from siphonnet.readout import ALPHAS

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRAIN, TEST = 40, 20  # samples per letter


def run_job(job):
    model, n, seed, rows = job
    rng = np.random.default_rng(seed)
    X_tr, y_tr = dataset(TRAIN, rng)
    X_te, y_te = dataset(TEST, rng)
    if model == "linear on pixels":
        clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS)).fit(X_tr, y_tr)
        return model, n, seed, clf.score(X_te, y_te)
    layout = "gravity" if "gravity" in model else "random"
    m = build(n, seed=100 + seed, recurrent="W=0" not in model, layout=layout,
              rows=rows if layout == "gravity" else None).fit(X_tr, y_tr)
    return model, n, seed, float(np.mean(m.predict(X_te) == y_te))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[36, 64, 100, 144])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rows", type=int, default=None, help="rows of the gravity cascade (default: square)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    models = ["physical, gravity cascade", "physical, random graph (needs pumps)", "physical SiphonNet, W=0",
              "linear on pixels"]
    jobs = [(m, n, s, args.rows) for n in args.sizes for m in models for s in range(args.seeds)]
    acc: dict = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for model, n, seed, a in pool.map(run_job, jobs):
            acc.setdefault((model, n), []).append(a)
    fmt = lambda v: f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"
    lines = [
        f"# Letters A–Z (5×7 pixels, 3 flipped, ±20 % pump tolerance; {TRAIN} train / {TEST} test per letter, {args.seeds} seeds)"
        + (f", cascade with {args.rows} rows" if args.rows else ""),
        "",
        "Test accuracy in %. Features: pulse counts and mean levels in 3 windows of a 30 s trial. "
        "'linear on pixels' does not depend on N.",
        "",
        "| model | " + " | ".join(f"N={n}" for n in args.sizes) + " |",
        "|---|" + "---|" * len(args.sizes),
    ]
    for m in models:
        lines.append(f"| {m} | " + " | ".join(fmt(acc[(m, n)]) for n in args.sizes) + " |")
    text = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / ("letters.md" if not args.rows else f"letters_rows{args.rows}.md")).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
