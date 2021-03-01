import json
import math

import numpy as np
import onnxruntime as ort
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from export import average_path_length, export_isolation_forest, floor_float32
from telemetry import FEATURES, anomalous_samples, labeled_mix, normal_samples
from train import train


def test_generators_have_expected_shape_and_separation():
    rng = np.random.default_rng(0)
    n, a = normal_samples(500, rng), anomalous_samples(500, rng)
    assert n.shape == a.shape == (500, len(FEATURES))
    assert n.dtype == np.float32 and np.isfinite(n).all() and np.isfinite(a).all()
    x, y = labeled_mix(1000, 0.1, rng)
    assert y.sum() == 100 and x.shape == (1000, 6)


def test_average_path_length_matches_sklearn():
    for n in (0, 1, 2, 3, 10, 256, 512):
        assert average_path_length(n) == pytest.approx(float(_average_path_length(np.array([n]))[0]))


def test_floor_float32_never_rounds_up():
    for t in (0.1, 1 / 3, 123456.789, -2.2):
        f = floor_float32(t)
        assert f <= t and np.float32(f) == f


def test_onnx_export_matches_sklearn_decision_function():
    rng = np.random.default_rng(3)
    x, _ = labeled_mix(3000, 0.05, rng)
    model = IsolationForest(n_estimators=40, max_samples=128, random_state=3).fit(x)
    proto = export_isolation_forest(model, x.shape[1])
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    probe, _ = labeled_mix(2000, 0.2, np.random.default_rng(4))
    got = sess.run(["scores"], {"X": probe})[0].reshape(-1)
    np.testing.assert_allclose(got, model.decision_function(probe), atol=1e-5)
    # Single-row and batched evaluation agree.
    one = sess.run(["scores"], {"X": probe[:1]})[0].reshape(-1)[0]
    assert one == pytest.approx(got[0], abs=1e-6)


def test_export_rejects_feature_subsampling():
    x, _ = labeled_mix(500, 0.05, np.random.default_rng(0))
    model = IsolationForest(n_estimators=5, max_features=0.5, random_state=0).fit(x)
    with pytest.raises(ValueError):
        export_isolation_forest(model, x.shape[1])


def test_train_writes_loadable_artifacts_and_detects_anomalies(tmp_path):
    meta = train(tmp_path, seed=1, n_train=8000)
    on_disk = json.loads((tmp_path / "model.json").read_text())
    assert on_disk["features"] == FEATURES and on_disk["score_sign"] == -1.0
    assert meta["metrics"]["roc_auc"] > 0.95
    assert meta["metrics"]["onnx_vs_sklearn_max_abs_diff"] < 1e-5
    sess = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(11)
    good = -sess.run(["scores"], {"X": normal_samples(2000, rng)})[0].reshape(-1)
    bad = -sess.run(["scores"], {"X": anomalous_samples(2000, rng)})[0].reshape(-1)
    thr = meta["threshold"]
    assert (good > thr).mean() < 0.05       # low false-positive rate
    assert (bad > thr).mean() > 0.6         # most injected incidents are flagged
    assert math.isfinite(thr)
