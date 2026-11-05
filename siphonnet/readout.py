"""Trained readouts: the only part of a reservoir computer that learns."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 4, 15)


def classify(F_train: np.ndarray, y_train: np.ndarray, F_test: np.ndarray, y_test: np.ndarray) -> float:
    """Standardised features + ridge classifier with built-in alpha selection; test accuracy."""
    model = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=ALPHAS))
    model.fit(F_train, y_train)
    return float(model.score(F_test, y_test))


def memory_capacity(
    states: np.ndarray, u: np.ndarray, max_delay: int = 40, washout: int = 100, train_frac: float = 0.7
) -> tuple[float, np.ndarray]:
    """Linear short-term memory capacity (Jaeger 2002).

    For each delay k = 1..max_delay a linear readout of ``states[t]`` is trained to recall
    ``u[t - k]``; the capacity is the sum of squared test correlations. It is bounded by
    the number of independent state features.
    """
    T = len(u)
    t = np.arange(washout, T)
    split = washout + int(train_frac * (T - washout))
    train, test = t[t < split], t[t >= split]
    scaler = StandardScaler().fit(states[train])
    X_train, X_test = scaler.transform(states[train]), scaler.transform(states[test])
    r2 = np.zeros(max_delay)
    for k in range(1, max_delay + 1):
        model = RidgeCV(alphas=ALPHAS).fit(X_train, u[train - k])
        pred = model.predict(X_test)
        c = np.corrcoef(pred, u[test - k])[0, 1]
        r2[k - 1] = 0.0 if not np.isfinite(c) else c**2
    return float(r2.sum()), r2
