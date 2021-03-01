"""Train an IsolationForest on synthetic telemetry and export it to ONNX.

Writes <out>/model.onnx and <out>/model.json (feature order, calibrated
threshold, evaluation metrics) which the C++ service loads at start-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from export import export_isolation_forest
from telemetry import FEATURES, labeled_mix, normal_samples

ONNX_OPSET = 17
N_ESTIMATORS = 300
MAX_SAMPLES = 512
ONNX_IR_VERSION = 9  # old enough for every ONNX Runtime >= 1.16


def train(out_dir: Path, seed: int = 7, n_train: int = 50_000, contamination: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    # Unsupervised: mostly-normal data with a little contamination, like real telemetry.
    x_train, _ = labeled_mix(n_train, contamination, rng)
    model = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, contamination=contamination,
                            random_state=seed, n_jobs=-1).fit(x_train)

    onx = export_isolation_forest(model, len(FEATURES), ONNX_OPSET, ONNX_IR_VERSION)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    score_idx = out_names.index("scores")

    def onnx_scores(x):
        return sess.run(None, {in_name: x.astype(np.float32)})[score_idx].reshape(-1)

    # ONNX "scores" output == sklearn decision_function (negative => outlier, lower => more
    # anomalous). The service flips the sign so higher == more anomalous.
    probe = x_train[:2000]
    max_diff = float(np.max(np.abs(onnx_scores(probe) - model.decision_function(probe))))
    assert max_diff < 1e-5, f"ONNX/sklearn mismatch: {max_diff}"

    # Calibrate threshold on clean data so the false-positive rate on clean data is ~1%.
    calib = normal_samples(20_000, np.random.default_rng(seed + 1))
    threshold = float(-np.quantile(onnx_scores(calib), 0.01))  # anomaly if -raw > threshold

    x_val, y_val = labeled_mix(20_000, 0.05, np.random.default_rng(seed + 2))
    anomaly_score = -onnx_scores(x_val)
    pred = (anomaly_score > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary")
    meta = {
        "name": "edgescore-isolation-forest",
        "features": FEATURES,
        "input_name": in_name,
        "score_output": "scores",
        "score_sign": -1.0,
        "threshold": threshold,
        "metrics": {
            "roc_auc": float(roc_auc_score(y_val, anomaly_score)),
            "avg_precision": float(average_precision_score(y_val, anomaly_score)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "onnx_vs_sklearn_max_abs_diff": max_diff,
        },
        "training": {"seed": seed, "n_train": n_train, "contamination": contamination,
                     "n_estimators": N_ESTIMATORS, "max_samples": MAX_SAMPLES},
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    m = train(args.out, args.seed)
    print(json.dumps(m["metrics"], indent=2))
    print(f"wrote {args.out}/model.onnx and model.json (threshold={m['threshold']:.4f})")
