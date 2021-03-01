"""Compact ONNX export for sklearn's IsolationForest.

skl2onnx emits one sub-graph per tree (~27 nodes/tree -> 8k nodes for 300 trees) which costs
milliseconds per call in ONNX Runtime. Here all trees go into a *single* TreeEnsembleRegressor
whose leaf value is `depth + c(n_node_samples)` (the isolation-forest path length), averaged
over the ensemble, followed by the closed-form score of the paper:

    decision_function(x) = -2 ** (-E[h(x)] / c(max_samples)) - offset_

The output named "scores" is therefore identical to IsolationForest.decision_function:
negative => outlier, lower => more anomalous.
"""
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import TensorProto, helper
from sklearn.ensemble import IsolationForest

EULER_GAMMA = 0.5772156649015329


def average_path_length(n: float) -> float:
    """c(n): expected path length of an unsuccessful BST search (matches sklearn)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


def floor_float32(t: float) -> float:
    """Largest float32 <= t, so that `x32 <= t32` agrees with sklearn's `x32 <= t64` for every
    float32 input x (ONNX stores thresholds as float32)."""
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def export_isolation_forest(model: IsolationForest, n_features: int, opset: int = 17,
                            ir_version: int = 9) -> onnx.ModelProto:
    n_trees = len(model.estimators_)
    attrs: dict[str, list] = {k: [] for k in (
        "nodes_treeids", "nodes_nodeids", "nodes_featureids", "nodes_modes", "nodes_values",
        "nodes_truenodeids", "nodes_falsenodeids", "target_ids", "target_nodeids",
        "target_treeids", "target_weights")}

    for tree_id, (est, feats) in enumerate(zip(model.estimators_, model.estimators_features_)):
        if len(feats) != n_features or not np.array_equal(feats, np.arange(n_features)):
            raise ValueError("feature sub-sampling (max_features < 1.0) is not supported")
        t = est.tree_
        # Node depths by walking from the root (sklearn numbers nodes in pre-order).
        depth = np.zeros(t.node_count, dtype=np.int64)
        for node in range(t.node_count):
            for child in (t.children_left[node], t.children_right[node]):
                if child != -1:
                    depth[child] = depth[node] + 1
        for node in range(t.node_count):
            leaf = t.children_left[node] == -1
            attrs["nodes_treeids"].append(tree_id)
            attrs["nodes_nodeids"].append(node)
            attrs["nodes_featureids"].append(0 if leaf else int(t.feature[node]))
            attrs["nodes_modes"].append("LEAF" if leaf else "BRANCH_LEQ")
            attrs["nodes_values"].append(0.0 if leaf else floor_float32(float(t.threshold[node])))
            attrs["nodes_truenodeids"].append(0 if leaf else int(t.children_left[node]))
            attrs["nodes_falsenodeids"].append(0 if leaf else int(t.children_right[node]))
            if leaf:
                attrs["target_ids"].append(0)
                attrs["target_nodeids"].append(node)
                attrs["target_treeids"].append(tree_id)
                attrs["target_weights"].append(
                    float(depth[node] + average_path_length(t.n_node_samples[node])))

    c_norm = average_path_length(model.max_samples_)
    tree_node = helper.make_node(
        "TreeEnsembleRegressor", ["X"], ["mean_path"], domain="ai.onnx.ml", n_targets=1,
        aggregate_function="AVERAGE", post_transform="NONE", **attrs)
    const = lambda name, v: helper.make_tensor(name, TensorProto.FLOAT, [1], [v])  # noqa: E731
    graph = helper.make_graph(
        [tree_node,
         helper.make_node("Mul", ["mean_path", "neg_ln2_over_c"], ["exponent"]),
         helper.make_node("Exp", ["exponent"], ["anomaly_raw"]),      # 2 ** (-E[h]/c)
         helper.make_node("Neg", ["anomaly_raw"], ["score_samples"]),  # sklearn score_samples
         helper.make_node("Sub", ["score_samples", "offset"], ["scores"])],
        "isolation_forest",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 1])],
        initializer=[const("neg_ln2_over_c", -math.log(2.0) / c_norm), const("offset", float(model.offset_))])
    proto = helper.make_model(
        graph, producer_name="edgescore",
        opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid("ai.onnx.ml", 3)])
    proto.ir_version = ir_version
    onnx.checker.check_model(proto)
    return proto
