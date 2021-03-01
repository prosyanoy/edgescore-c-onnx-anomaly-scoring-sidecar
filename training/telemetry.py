"""Synthetic host telemetry generator shared by training, tests and benchmarks.

Normal behaviour is a correlated multivariate process (traffic drives CPU,
latency and egress); anomalies are injected as four realistic failure modes.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["cpu_pct", "mem_pct", "latency_ms", "error_rate", "req_rate", "bytes_out_kb"]


def normal_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    req = rng.normal(200.0, 30.0, n).clip(20, None)
    cpu = (18.0 + 0.16 * req + rng.normal(0, 3.0, n)).clip(1, 99)
    mem = (45.0 + rng.normal(0, 5.0, n)).clip(5, 95)
    latency = (35.0 + 0.10 * cpu + rng.gamma(4.0, 3.0, n))
    err = rng.beta(1.0, 400.0, n) * 100.0  # percent, ~0.25 on average
    out = (req * 4.0 + rng.normal(0, 25.0, n)).clip(1, None)
    return np.column_stack([cpu, mem, latency, err, req, out]).astype(np.float32)


def anomalous_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    x = normal_samples(n, rng)
    kind = rng.integers(0, 4, n)
    cpu, mem, lat, err, req, out = (x[:, i].copy() for i in range(6))
    m = kind == 0  # CPU saturation
    cpu[m] = rng.uniform(92, 100, m.sum())
    lat[m] *= rng.uniform(2.0, 4.0, m.sum())
    m = kind == 1  # downstream failure: latency + errors
    lat[m] = rng.uniform(300, 1500, m.sum())
    err[m] = rng.uniform(8, 40, m.sum())
    m = kind == 2  # traffic collapse
    req[m] = rng.uniform(0, 25, m.sum())
    out[m] *= 0.05
    m = kind == 3  # memory leak
    mem[m] = rng.uniform(90, 99, m.sum())
    return np.column_stack([cpu, mem, lat, err, req, out]).astype(np.float32)


def labeled_mix(n: int, anomaly_rate: float, rng: np.random.Generator):
    n_bad = int(n * anomaly_rate)
    x = np.vstack([normal_samples(n - n_bad, rng), anomalous_samples(n_bad, rng)])
    y = np.concatenate([np.zeros(n - n_bad, dtype=int), np.ones(n_bad, dtype=int)])
    idx = rng.permutation(n)
    return x[idx], y[idx]
