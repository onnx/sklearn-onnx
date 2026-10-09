# SPDX-License-Identifier: Apache-2.0

"""
Common functions to convert any learner based on trees.
"""

import numpy as np
from onnx.helper import np_dtype_to_tensor_dtype
from onnx.numpy_helper import from_array
from ..proto import onnx_proto
from ._topology import get_default_opset_for_domain
from .data_types import DoubleTensorType, guess_numpy_type


def get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["class_treeids"] = []
    attrs["class_nodeids"] = []
    attrs["class_ids"] = []
    attrs["class_weights"] = []
    return attrs


def get_default_tree_regressor_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["n_targets"] = 0
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["target_treeids"] = []
    attrs["target_nodeids"] = []
    attrs["target_ids"] = []
    attrs["target_weights"] = []
    return attrs


def find_switch_point(fy, nfy):
    """
    Finds the double so that
    ``(float)x != (float)(x + espilon)``.
    """
    a = np.float64(fy)
    b = np.float64(nfy)
    fa = np.float32(a)
    a0, b0 = a, a
    while a != a0 or b != b0:
        a0, b0 = a, b
        m = (a + b) / 2
        fm = np.float32(m)
        if fm == fa:
            a = m
            fa = fm
        else:
            b = m
    return a


def sklearn_threshold(dy, dtype, mode):
    """
    *scikit-learn* does not compare x to a threshold
    but (float)x to a double threshold. As we need a float
    threshold, we need a different value than the threshold
    rounded to float. For floats, it finds float *w* which
    verifies::

        (float)x <= y <=> (float)x <= w

    For doubles, it finds double *w* which verifies::

        (float)x <= y <=> x <= w
    """
    if mode == "BRANCH_LEQ":
        if dtype == np.float32:
            fy = np.float32(dy)
            if fy == dy:
                return np.float64(fy)
            if fy < dy:
                return np.float64(fy)
            eps = max(abs(fy), np.finfo(np.float32).eps) * 10
            nfy = np.nextafter([fy], [fy - eps], dtype=np.float32)[0]
            return np.float64(nfy)
        elif dtype == np.float64:
            fy = np.float32(dy)
            eps = max(abs(fy), np.finfo(np.float32).eps) * 10
            afy = np.nextafter([fy], [fy - eps], dtype=np.float32)[0]
            afy2 = find_switch_point(afy, fy)
            if fy > dy > afy2:
                return afy2
            bfy = np.nextafter([fy], [fy + eps], dtype=np.float32)[0]
            bfy2 = find_switch_point(fy, bfy)
            if fy <= dy <= bfy2:
                return bfy2
            return np.float64(fy)
        raise TypeError("Unexpected dtype {}.".format(dtype))
    raise RuntimeError(
        "Threshold is not changed for other mode and "
        "'BRANCH_LEQ' (actually '{}').".format(mode)
    )


def add_node(
    attr_pairs,
    is_classifier,
    tree_id,
    tree_weight,
    node_id,
    feature_id,
    mode,
    value,
    true_child_id,
    false_child_id,
    weights,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn,
    dtype,
    nodes_missing_value_tracks_true=0,  # cannot use boolean in onnx
):
    attr_pairs["nodes_treeids"].append(tree_id)
    attr_pairs["nodes_nodeids"].append(node_id)
    attr_pairs["nodes_featureids"].append(feature_id)
    attr_pairs["nodes_modes"].append(mode)
    if adjust_threshold_for_sklearn and mode != "LEAF":
        attr_pairs["nodes_values"].append(sklearn_threshold(value, dtype, mode))
    else:
        attr_pairs["nodes_values"].append(value)
    attr_pairs["nodes_truenodeids"].append(true_child_id)
    attr_pairs["nodes_falsenodeids"].append(false_child_id)
    attr_pairs["nodes_missing_value_tracks_true"].append(
        int(nodes_missing_value_tracks_true)  # cannot use boolean in onnx
    )
    attr_pairs["nodes_hitrates"].append(1.0)

    # Add leaf information for making prediction
    if mode == "LEAF":
        flattened_weights = weights.flatten()
        factor = tree_weight
        # If the values stored at leaves are counts of possible classes, we
        # need convert them to probabilities by doing a normalization.
        if leaf_weights_are_counts:
            s = sum(flattened_weights)
            factor /= float(s) if s != 0.0 else 1.0
        flattened_weights = [w * factor for w in flattened_weights]
        if len(flattened_weights) == 2 and is_classifier:
            flattened_weights = [flattened_weights[1]]

        # Note that attribute names for making prediction are different for
        # classifiers and regressors
        if is_classifier:
            for i, w in enumerate(flattened_weights):
                attr_pairs["class_treeids"].append(tree_id)
                attr_pairs["class_nodeids"].append(node_id)
                attr_pairs["class_ids"].append(i + weight_id_bias)
                attr_pairs["class_weights"].append(w)
        else:
            for i, w in enumerate(flattened_weights):
                attr_pairs["target_treeids"].append(tree_id)
                attr_pairs["target_nodeids"].append(node_id)
                attr_pairs["target_ids"].append(i + weight_id_bias)
                attr_pairs["target_weights"].append(w)


def add_tree_to_attribute_pairs(
    attr_pairs,
    is_classifier,
    tree,
    tree_id,
    tree_weight,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn=False,
    dtype=None,
):
    # scikit-learn >= 1.3/1.4 trees route NaN per-node via missing_go_to_left,
    # learned at fit time; trees fitted by older sklearn lack the attribute.
    missing_go_to_left = getattr(tree, "missing_go_to_left", None)
    for i in range(tree.node_count):
        node_id = i
        weight = tree.value[i]

        if tree.children_left[i] > i or tree.children_right[i] > i:
            mode = "BRANCH_LEQ"
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
            missing = (
                int(missing_go_to_left[i]) if missing_go_to_left is not None else 0
            )
        else:
            mode = "LEAF"
            feat_id = 0
            threshold = 0.0
            left_child_id = 0
            right_child_id = 0
            missing = 0

        add_node(
            attr_pairs,
            is_classifier,
            tree_id,
            tree_weight,
            node_id,
            feat_id,
            mode,
            threshold,
            left_child_id,
            right_child_id,
            weight,
            weight_id_bias,
            leaf_weights_are_counts,
            adjust_threshold_for_sklearn=adjust_threshold_for_sklearn,
            dtype=dtype,
            nodes_missing_value_tracks_true=missing,
        )


def add_tree_to_attribute_pairs_hist_gradient_boosting(
    attr_pairs,
    is_classifier,
    tree,
    tree_id,
    tree_weight,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn=False,
    dtype=None,
):
    for i, node in enumerate(tree.nodes):
        node_id = i
        weight = node["value"]

        if node["is_leaf"]:
            mode = "LEAF"
            feat_id = 0
            threshold = 0.0
            left_child_id = 0
            right_child_id = 0
            missing = False
        else:
            mode = "BRANCH_LEQ"
            feat_id = node["feature_idx"]
            try:
                threshold = node["threshold"]
            except ValueError:
                threshold = node["num_threshold"]
            left_child_id = node["left"]
            right_child_id = node["right"]
            missing = node["missing_go_to_left"]

        add_node(
            attr_pairs,
            is_classifier,
            tree_id,
            tree_weight,
            node_id,
            feat_id,
            mode,
            threshold,
            left_child_id,
            right_child_id,
            weight,
            weight_id_bias,
            leaf_weights_are_counts,
            adjust_threshold_for_sklearn=adjust_threshold_for_sklearn,
            dtype=dtype,
            nodes_missing_value_tracks_true=missing,
        )


_TREE_ENSEMBLE_MODES = {
    "BRANCH_LEQ": 0,
    "BRANCH_LT": 1,
    "BRANCH_GTE": 2,
    "BRANCH_GT": 3,
    "BRANCH_EQ": 4,
    "BRANCH_NEQ": 5,
}


def uses_tree_ensemble(container):
    """
    Tells if the target opset of domain *ai.onnx.ml*, given by the user or
    the default one, is 5 or more and allows operator *TreeEnsemble*.
    """
    opv = container.target_opset_all.get("ai.onnx.ml") or get_default_opset_for_domain(
        "ai.onnx.ml"
    )
    return opv >= 5


def convert_to_tree_ensemble_attributes(attrs, dtype):
    """
    Converts the attributes of a *TreeEnsembleClassifier* or a
    *TreeEnsembleRegressor* into the attributes of operator
    *TreeEnsemble* (domain *ai.onnx.ml*, opset 5).

    Every leaf of *TreeEnsemble* contributes to one target: a tree whose
    leaves hold a weight for several targets is written once per target.

    :param attrs: attributes of the deprecated operator
    :param dtype: numpy.float32 or numpy.float64, type of the thresholds
        and the leaf weights
    :return: attributes of operator *TreeEnsemble*
    """
    prefix = "class_" if "class_ids" in attrs else "target_"
    leaf_values = {}
    for tree_id, node_id, target, weight in zip(
        attrs[prefix + "treeids"],
        attrs[prefix + "nodeids"],
        attrs[prefix + "ids"],
        attrs[prefix + "weights"],
    ):
        leaf_values.setdefault((int(tree_id), int(node_id)), []).append(
            (int(target), float(weight))
        )

    modes = attrs["nodes_modes"]
    trees = {}
    for k, (tree_id, node_id) in enumerate(
        zip(attrs["nodes_treeids"], attrs["nodes_nodeids"])
    ):
        trees.setdefault(int(tree_id), {})[int(node_id)] = k

    aggregate = attrs.get("aggregate_function", "SUM")
    if aggregate == "AVERAGE":
        # A tree may be written once per target, the average is a scaled sum.
        scale = 1.0 / len(trees)
    elif aggregate == "SUM":
        scale = 1.0
    else:
        raise NotImplementedError(
            f"aggregate_function={aggregate!r} cannot be converted into "
            f"operator TreeEnsemble."
        )

    new_attrs = {
        "nodes_featureids": [],
        "nodes_splits": [],
        "nodes_modes": [],
        "nodes_truenodeids": [],
        "nodes_trueleafs": [],
        "nodes_falsenodeids": [],
        "nodes_falseleafs": [],
        "nodes_missing_value_tracks_true": [],
        "leaf_targetids": [],
        "leaf_weights": [],
        "tree_roots": [],
    }

    def add_leaf(target, weight):
        new_attrs["leaf_targetids"].append(target)
        new_attrs["leaf_weights"].append(weight)
        return len(new_attrs["leaf_targetids"]) - 1

    def add_split(feature, split, mode, missing):
        new_attrs["nodes_featureids"].append(feature)
        new_attrs["nodes_splits"].append(split)
        new_attrs["nodes_modes"].append(mode)
        new_attrs["nodes_missing_value_tracks_true"].append(missing)
        for k in ("truenodeids", "trueleafs", "falsenodeids", "falseleafs"):
            new_attrs["nodes_" + k].append(0)
        return len(new_attrs["nodes_featureids"]) - 1

    def set_child(index, branch, child_id, is_leaf):
        new_attrs[f"nodes_{branch}nodeids"][index] = child_id
        new_attrs[f"nodes_{branch}leafs"][index] = int(is_leaf)

    def add_tree(nodes, root, leaf_of):
        # leaf_of(node_id) returns the (target, weight) of a leaf.
        if modes[nodes[root]] == "LEAF":
            # A tree reduced to one leaf is a node whose branches
            # both point to that leaf.
            leaf = add_leaf(*leaf_of(root))
            index = add_split(0, 0.0, 0, 0)
            set_child(index, "true", leaf, True)
            set_child(index, "false", leaf, True)
            new_attrs["tree_roots"].append(index)
            return
        index_of = {}
        stack = [root]
        while stack:
            node_id = stack.pop()
            k = nodes[node_id]
            mode = modes[k]
            if mode not in _TREE_ENSEMBLE_MODES:
                raise NotImplementedError(
                    f"Mode {mode!r} cannot be converted into operator TreeEnsemble."
                )
            index_of[node_id] = add_split(
                int(attrs["nodes_featureids"][k]),
                attrs["nodes_values"][k],
                _TREE_ENSEMBLE_MODES[mode],
                int(attrs["nodes_missing_value_tracks_true"][k]),
            )
            for child in (
                attrs["nodes_falsenodeids"][k],
                attrs["nodes_truenodeids"][k],
            ):
                if modes[nodes[int(child)]] != "LEAF":
                    stack.append(int(child))
        for node_id, index in index_of.items():
            k = nodes[node_id]
            for branch in ("true", "false"):
                child = int(attrs[f"nodes_{branch}nodeids"][k])
                if modes[nodes[child]] == "LEAF":
                    set_child(index, branch, add_leaf(*leaf_of(child)), True)
                else:
                    set_child(index, branch, index_of[child], False)
        root_index = index_of[root]
        # onnxruntime < 1.24 reads a root as a leaf when its two branches
        # have the same index, one being a leaf and the other one a node
        # (microsoft/onnxruntime#24679). The leaf is copied until both differ.
        while (
            new_attrs["nodes_truenodeids"][root_index]
            == new_attrs["nodes_falsenodeids"][root_index]
            and new_attrs["nodes_trueleafs"][root_index]
            != new_attrs["nodes_falseleafs"][root_index]
        ):
            branch = "true" if new_attrs["nodes_trueleafs"][root_index] else "false"
            leaf = new_attrs[f"nodes_{branch}nodeids"][root_index]
            copy = add_leaf(
                new_attrs["leaf_targetids"][leaf], new_attrs["leaf_weights"][leaf]
            )
            set_child(root_index, branch, copy, True)
        new_attrs["tree_roots"].append(root_index)

    for tree_id in sorted(trees):
        nodes = trees[tree_id]
        children = set()
        for k in nodes.values():
            if modes[k] != "LEAF":
                children.add(int(attrs["nodes_truenodeids"][k]))
                children.add(int(attrs["nodes_falsenodeids"][k]))
        roots = [node_id for node_id in nodes if node_id not in children]
        if len(roots) != 1:
            raise RuntimeError(f"Tree {tree_id} has {len(roots)} roots, expected 1.")
        values = {
            node_id: leaf_values.get((tree_id, node_id), [])
            for node_id in nodes
            if modes[nodes[node_id]] == "LEAF"
        }
        if all(len(v) <= 1 for v in values.values()):
            add_tree(
                nodes,
                roots[0],
                lambda node_id, values=values: (
                    values[node_id][0] if values[node_id] else (0, 0.0)
                ),
            )
            continue
        for target in sorted({t for v in values.values() for t, _ in v}):
            add_tree(
                nodes,
                roots[0],
                lambda node_id, target=target, values=values: (
                    target,
                    sum(w for t, w in values[node_id] if t == target),
                ),
            )

    new_attrs["nodes_splits"] = from_array(
        np.array(new_attrs["nodes_splits"], dtype=dtype)
    )
    new_attrs["nodes_modes"] = from_array(
        np.array(new_attrs["nodes_modes"], dtype=np.uint8)
    )
    new_attrs["leaf_weights"] = from_array(
        (np.array(new_attrs["leaf_weights"]) * scale).astype(dtype)
    )
    new_attrs["aggregate_function"] = 1
    new_attrs["post_transform"] = 0
    return new_attrs


def add_tree_ensemble_node(
    scope,
    container,
    op_type,
    input_name,
    output_names,
    dtype,
    op_domain="ai.onnx.ml",
    op_version=1,
    **attrs,
):
    """
    Adds a *TreeEnsembleClassifier* or a *TreeEnsembleRegressor*. If the
    target opset of domain *ai.onnx.ml* is 5 or more, adds instead operator
    *TreeEnsemble* followed by the operators computing the outputs of the
    deprecated operator.

    :param scope: scope
    :param container: container
    :param op_type: *TreeEnsembleClassifier* or *TreeEnsembleRegressor*
    :param input_name: input name
    :param output_names: label and scores for a classifier,
        predictions for a regressor
    :param dtype: numpy.float32 or numpy.float64, type of the thresholds,
        the leaf weights and the scores
    :param op_domain: domain of the deprecated operator
    :param op_version: opset of the deprecated operator
    :param attrs: attributes of the deprecated operator
    """
    if not uses_tree_ensemble(container):
        container.add_node(
            op_type,
            input_name,
            output_names,
            op_domain=op_domain,
            op_version=op_version,
            **attrs,
        )
        return

    if isinstance(input_name, list):
        (input_name,) = input_name
    if isinstance(output_names, str):
        output_names = [output_names]
    proto_dtype = np_dtype_to_tensor_dtype(np.dtype(dtype))
    name = attrs.get("name", op_type)
    base_values = list(attrs.get("base_values", []))
    post_transform = attrs.get("post_transform", "NONE")
    if post_transform not in ("NONE", "LOGISTIC", "SOFTMAX"):
        raise NotImplementedError(
            f"post_transform={post_transform!r} cannot be converted into "
            f"operator TreeEnsemble."
        )

    def add(op, inputs, output=None, **kwargs):
        if output is None:
            output = scope.get_unique_variable_name(f"{name}_{op}")
        container.add_node(
            op, inputs, output, name=scope.get_unique_operator_name(op), **kwargs
        )
        return output

    def constant(values, proto_type=proto_dtype):
        values = np.asarray(values)
        constant_name = scope.get_unique_variable_name(f"{name}_cst")
        container.add_initializer(
            constant_name, proto_type, list(values.shape), values.ravel().tolist()
        )
        return constant_name

    def write(source, output):
        # An output declared double stays double, any other one is float
        # as with the deprecated operator.
        variable = scope.get(output, None)
        out_dtype = (
            np.float64
            if variable is not None and isinstance(variable.type, DoubleTensorType)
            else np.float32
        )
        if out_dtype == dtype:
            add("Identity", [source], output)
        else:
            add(
                "Cast",
                [source],
                output,
                to=np_dtype_to_tensor_dtype(np.dtype(out_dtype)),
            )

    def tree_ensemble(n_targets):
        scores = add(
            "TreeEnsemble",
            [input_name],
            op_domain="ai.onnx.ml",
            op_version=5,
            n_targets=n_targets,
            **convert_to_tree_ensemble_attributes(attrs, dtype),
        )
        if base_values:
            scores = add("Add", [scores, constant(base_values)])
        return scores

    def transform(scores):
        if post_transform == "LOGISTIC":
            return add("Sigmoid", [scores])
        if post_transform == "SOFTMAX":
            return add("Softmax", [scores], axis=1)
        return scores

    # TreeEnsemble only accepts float inputs. An input the converter
    # created itself is already cast.
    variable = scope.get(input_name, None)
    if variable is not None and guess_numpy_type(variable.type) != dtype:
        input_name = add("Cast", [input_name], to=proto_dtype)

    if op_type == "TreeEnsembleRegressor":
        write(transform(tree_ensemble(int(attrs["n_targets"]))), output_names[0])
        return

    if "classlabels_strings" in attrs:
        labels = constant(
            np.array([str(s).encode("utf-8") for s in attrs["classlabels_strings"]]),
            onnx_proto.TensorProto.STRING,
        )
        n_classes = len(attrs["classlabels_strings"])
    else:
        labels = constant(
            np.array(attrs["classlabels_int64s"], dtype=np.int64),
            onnx_proto.TensorProto.INT64,
        )
        n_classes = len(attrs["classlabels_int64s"])
    class_ids = {int(i) for i in attrs["class_ids"]}
    label_output, scores_output = output_names

    if n_classes == 2:
        # The trees compute the score s of the second class. As the deprecated
        # operator: if all weights are positive, the label is the second class
        # when s > 0.5 and the scores are [1 - s, s], otherwise when s > 0 and
        # the scores are [-s, s]. LOGISTIC gives [sigmoid(-s), sigmoid(s)].
        if class_ids != {0} or len(base_values) > 1 or post_transform == "SOFTMAX":
            raise NotImplementedError(
                f"A binary classifier with class_ids={class_ids}, "
                f"base_values={base_values} and post_transform={post_transform!r} "
                f"cannot be converted into operator TreeEnsemble."
            )
        scores = tree_ensemble(1)
        positive = all(float(w) >= 0 for w in attrs["class_weights"])
        if post_transform == "LOGISTIC":
            proba = [add("Sigmoid", [add("Neg", [scores])]), add("Sigmoid", [scores])]
        elif positive:
            proba = [add("Sub", [constant([1.0]), scores]), scores]
        else:
            proba = [add("Neg", [scores]), scores]
        write(add("Concat", proba, axis=1), scores_output)
        index = add(
            "Cast",
            [add("Greater", [scores, constant([0.5 if positive else 0.0])])],
            to=onnx_proto.TensorProto.INT64,
        )
        index = add("Reshape", [index, constant([-1], onnx_proto.TensorProto.INT64)])
    else:
        # The label is the class of highest score among the classes
        # the leaves or the base values score.
        if base_values and len(base_values) != n_classes:
            raise NotImplementedError(
                f"base_values must have {n_classes} values not {len(base_values)}."
            )
        scores = tree_ensemble(n_classes)
        write(transform(scores), scores_output)
        absent = [i for i in range(n_classes) if i not in class_ids]
        if absent and not base_values:
            mask = np.zeros(n_classes, dtype=dtype)
            mask[absent] = -np.inf
            scores = add("Add", [scores, constant(mask)])
        index = add("ArgMax", [scores], axis=1, keepdims=0)
    add("Gather", [labels, index], label_output, axis=0)
