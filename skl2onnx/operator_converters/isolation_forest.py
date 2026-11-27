# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ..common._registration import register_converter
from ..common.data_types import FloatTensorType, guess_numpy_type
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
)
from ..proto import onnx_proto
from ..algebra.onnx_ops import (
    OnnxTreeEnsembleRegressor_1,
    OnnxCast,
    OnnxLess,
    OnnxMul,
    OnnxAdd,
    OnnxNeg,
    OnnxPow,
)


def _average_path_length(n_samples):
    """
    Average path length of an unsuccessful search in a binary search
    tree built from *n_samples* points, as defined by scikit-learn's
    ``IsolationForest``: 0 for n <= 1, 1 for n == 2 and
    ``2 * (log(n - 1) + euler_gamma) - 2 * (n - 1) / n`` otherwise.
    """
    n = np.asarray(n_samples, dtype=np.float64)
    res = np.zeros(n.shape, dtype=np.float64)
    res[n == 2] = 1.0
    big = n > 2
    res[big] = (
        2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    )
    return res


def _node_depths(tree):
    """Depth of every node, the root being at depth 1 as in scikit-learn."""
    depths = np.zeros(tree.node_count, dtype=np.float64)
    depths[0] = 1.0
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (tree.children_left[node], tree.children_right[node]):
            if child != -1:
                depths[child] = depths[node] + 1.0
                stack.append(child)
    return depths


def convert_sklearn_isolation_forest(
    scope,
    operator,
    container,
    op_type="TreeEnsembleRegressor",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    """
    Converts an *IsolationForest*.

    For every sample, scikit-learn sums over all trees the depth of the
    leaf the sample falls into, adjusted by the average path length of the
    training samples left in that leaf. That value only depends on the
    leaf, so it is computed here once per leaf and stored as the leaf
    weight of a single *TreeEnsembleRegressor* which sums them over all
    trees. The graph size therefore does not grow with the number of
    trees.
    """
    op = operator.raw_operator
    outputs = operator.outputs
    opv = container.target_opset
    if opv < 7:
        # Add and Mul only support numpy-style broadcasting since opset 7.
        raise RuntimeError(
            "This converter requires at least opset 7 for the main domain, "
            "got %r." % opv
        )
    options = container.get_options(op, dict(score_samples=None))

    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    proto_dtype = (
        onnx_proto.TensorProto.DOUBLE
        if dtype == np.float64
        else onnx_proto.TensorProto.FLOAT
    )

    # scikit-learn evaluates the trees on a float32 copy of the features.
    input_name = operator.inputs[0]
    if not isinstance(operator.inputs[0].type, FloatTensorType):
        input_name = OnnxCast(
            input_name, to=onnx_proto.TensorProto.FLOAT, op_version=opv
        )

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs["n_targets"] = 1
    attrs["post_transform"] = "NONE"
    attrs["aggregate_function"] = "SUM"
    for tree_id, (estimator, features) in enumerate(
        zip(op.estimators_, op.estimators_features_)
    ):
        tree = estimator.tree_
        first_node = len(attrs["nodes_featureids"])
        first_leaf = len(attrs["target_nodeids"])
        add_tree_to_attribute_pairs(
            attrs, False, tree, tree_id, 1.0, 0, False, True, dtype=np.float32
        )
        # Each tree is trained on a subset of the features: map its feature
        # indices back to the columns of the input.
        features = np.asarray(features, dtype=np.int64)
        attrs["nodes_featureids"][first_node:] = [
            int(features[f]) for f in attrs["nodes_featureids"][first_node:]
        ]
        # Leaf weight: depth of the leaf + average path length of the
        # training samples in the leaf - 1, as in
        # IsolationForest._compute_score_samples.
        leaf_values = (
            _node_depths(tree) + _average_path_length(tree.n_node_samples) - 1.0
        )
        attrs["target_weights"][first_leaf:] = [
            float(leaf_values[n]) for n in attrs["target_nodeids"][first_leaf:]
        ]

    depth_sum = OnnxTreeEnsembleRegressor_1(input_name, op_version=1, **attrs)
    if dtype == np.float64:
        depth_sum = OnnxCast(depth_sum, to=proto_dtype, op_version=opv)

    # score_samples = -2 ** (-depth_sum / (n_estimators * c(max_samples)))
    norm = len(op.estimators_) * _average_path_length([op.max_samples_])[0]
    output_names = outputs[2].full_name if options["score_samples"] else None
    if norm == 0:
        # scikit-learn sets the normalized depth to zero when max_samples is 1.
        score_samples = OnnxAdd(
            depth_sum,
            np.array([-0.5], dtype=dtype),
            op_version=opv,
            output_names=output_names,
        )
    else:
        neg_depth = OnnxMul(
            depth_sum, np.array([-1.0 / norm], dtype=dtype), op_version=opv
        )
        score_samples = OnnxNeg(
            OnnxPow(np.array([2], dtype=dtype), neg_depth, op_version=opv),
            op_version=opv,
            output_names=output_names,
        )

    # decision_function = score_samples - offset_
    decision = OnnxAdd(
        score_samples,
        np.array([-op.offset_], dtype=dtype),
        op_version=opv,
        output_names=outputs[1].full_name,
    )
    decision.set_onnx_name_prefix("dec")

    # predict = -1 for outliers (decision < 0), 1 otherwise
    less = OnnxLess(decision, np.array([0], dtype=dtype), op_version=opv)
    predict = OnnxAdd(
        OnnxMul(
            OnnxCast(less, op_version=opv, to=onnx_proto.TensorProto.INT64),
            np.array([-2], dtype=np.int64),
            op_version=opv,
        ),
        np.array([1], dtype=np.int64),
        op_version=opv,
        output_names=outputs[0].full_name,
    )
    predict.set_onnx_name_prefix("predict")

    predict.add_to(scope, container)
    if options["score_samples"]:
        score_samples.add_to(scope, container)


register_converter(
    "SklearnIsolationForest",
    convert_sklearn_isolation_forest,
    options={"score_samples": [True, False]},
)
