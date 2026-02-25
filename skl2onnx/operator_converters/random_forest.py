# SPDX-License-Identifier: Apache-2.0


import numbers
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from ..common._apply_operation import (
    apply_add,
    apply_cast,
    apply_concat,
    apply_reshape,
    apply_transpose,
)
from ..common.data_types import BooleanTensorType, Int64TensorType, guess_numpy_type
from ..common._registration import register_converter
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    add_tree_to_attribute_pairs_hist_gradient_boosting,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
    get_left_categories_from_bitset,
)
from ..common.utils_classifier import get_label_classes
from ..proto import onnx_proto
from .decision_tree import (
    predict,
    _build_labels_path,
    _build_labels_leaf,
    _append_decision_output,
)


def _hgb_has_preprocessor(op):
    """Return True if the HGB model has an internal feature preprocessor.

    When ``categorical_features`` is specified, scikit-learn wraps the input
    with a :class:`~sklearn.compose.ColumnTransformer` that (a) OrdinalEncodes
    categorical columns and (b) places them *before* the numerical columns.
    The tree nodes then use feature indices in this *remapped* space, so the
    converter must replicate the same preprocessing.
    """
    return getattr(op, "_preprocessor", None) is not None


def _build_hgb_categorical_preproc(scope, container, operator, op, dtype):
    """Build the ONNX subgraph that replicates the HGB internal preprocessor.

    The preprocessor consists of:
    1. OrdinalEncoding each categorical column (maps known values to float
       indices 0, 1, ...; unknown values become NaN).
    2. Reordering: categorical columns first, then numerical columns.

    Returns the name of the preprocessed feature tensor (shape [N, n_features]).
    """
    preprocessor = op._preprocessor
    enc = preprocessor.named_transformers_["encoder"]
    is_categorical = op.is_categorical_  # bool array, length = n_features_in_
    cat_orig_indices = np.where(is_categorical)[0].tolist()
    num_orig_indices = np.where(~is_categorical)[0].tolist()

    input_name = operator.input_full_names[0]
    output_parts = []

    # ------------------------------------------------------------------
    # Encode each categorical column with a LabelEncoder
    # ------------------------------------------------------------------
    for enc_idx, orig_col in enumerate(cat_orig_indices):
        categories = enc.categories_[enc_idx].astype(dtype)

        # Extract column orig_col: Gather on axis=1
        col_idx_name = scope.get_unique_variable_name(f"cat_col_idx_{orig_col}")
        container.add_initializer(
            col_idx_name,
            onnx_proto.TensorProto.INT64,
            [],
            [orig_col],
        )
        col_raw_name = scope.get_unique_variable_name(f"cat_col_raw_{orig_col}")
        container.add_node(
            "Gather",
            [input_name, col_idx_name],
            col_raw_name,
            op_version=13,
            name=scope.get_unique_operator_name("Gather"),
            axis=1,
        )

        # Flatten to 1-D for LabelEncoder
        col_flat_name = scope.get_unique_variable_name(f"cat_col_flat_{orig_col}")
        apply_reshape(
            scope, col_raw_name, col_flat_name, container, desired_shape=(-1,)
        )

        # Apply LabelEncoder: known -> index; unknown -> NaN
        keys = categories.tolist()
        values = list(range(len(categories)))
        col_encoded_name = scope.get_unique_variable_name(f"cat_col_enc_{orig_col}")
        if dtype == np.float64:
            container.add_node(
                "LabelEncoder",
                [col_flat_name],
                col_encoded_name,
                op_domain="ai.onnx.ml",
                op_version=4,
                name=scope.get_unique_operator_name("LabelEncoder"),
                keys_doubles=keys,
                values_doubles=[float(v) for v in values],
                default_double=float("nan"),
            )
        else:
            container.add_node(
                "LabelEncoder",
                [col_flat_name],
                col_encoded_name,
                op_domain="ai.onnx.ml",
                op_version=4,
                name=scope.get_unique_operator_name("LabelEncoder"),
                keys_floats=keys,
                values_floats=[float(v) for v in values],
                default_float=float("nan"),
            )

        # Reshape back to [N, 1]
        col_2d_name = scope.get_unique_variable_name(f"cat_col_2d_{orig_col}")
        apply_reshape(
            scope, col_encoded_name, col_2d_name, container, desired_shape=(-1, 1)
        )
        output_parts.append(col_2d_name)

    # ------------------------------------------------------------------
    # Extract each numerical column (no transformation needed)
    # ------------------------------------------------------------------
    for orig_col in num_orig_indices:
        col_idx_name = scope.get_unique_variable_name(f"num_col_idx_{orig_col}")
        container.add_initializer(
            col_idx_name,
            onnx_proto.TensorProto.INT64,
            [],
            [orig_col],
        )
        col_name = scope.get_unique_variable_name(f"num_col_{orig_col}")
        container.add_node(
            "Gather",
            [input_name, col_idx_name],
            col_name,
            op_version=13,
            name=scope.get_unique_operator_name("Gather"),
            axis=1,
        )

        # Reshape to [N, 1]
        col_2d_name = scope.get_unique_variable_name(f"num_col_2d_{orig_col}")
        apply_reshape(scope, col_name, col_2d_name, container, desired_shape=(-1, 1))
        output_parts.append(col_2d_name)

    # ------------------------------------------------------------------
    # Concatenate: [cat_cols..., num_cols...]
    # ------------------------------------------------------------------
    if len(output_parts) == 1:
        return output_parts[0]

    preprocessed_name = scope.get_unique_variable_name("hgb_preprocessed")
    apply_concat(scope, output_parts, preprocessed_name, container, axis=1)
    return preprocessed_name


def _build_hgb_tree_ensemble(
    scope, container, op, preprocessed_name, dtype, n_targets=1
):
    """Build a ``TreeEnsemble`` (ai.onnx.ml opset 5) node for an HGB predictor.

    This supports both numerical splits (BRANCH_LEQ) and categorical splits
    (BRANCH_MEMBER) using ``raw_left_cat_bitsets``.

    Parameters
    ----------
    n_targets : int
        Number of output targets (1 for regressor/binary, n_classes for
        multi-class classifier).

    Returns the name of the raw-score output tensor (shape [N, n_targets]).
    """
    # ----------------------------------------------------------------
    # Collect all-trees arrays for the TreeEnsemble op
    # ----------------------------------------------------------------
    all_nodes_modes = []
    all_nodes_featureids = []
    all_nodes_splits = []
    all_nodes_truenodeids = []
    all_nodes_falsenodeids = []
    all_nodes_trueleafs = []
    all_nodes_falseleafs = []
    all_nodes_hitrates = []
    all_nodes_missing_value_tracks_true = []
    all_leaf_weights = []
    all_leaf_targetids = []
    membership_values_list = []  # one list per BRANCH_MEMBER node, in node-index order
    tree_roots = []

    global_split_offset = 0
    global_leaf_offset = 0

    for predictors_list in op._predictors:
        for target_id, tree in enumerate(predictors_list):
            nodes = tree.nodes
            raw_left_cat_bitsets = tree.raw_left_cat_bitsets

            # --------------------------------------------------------------
            # Pass 1: assign split/leaf indices
            # orig_idx -> (is_leaf, local_split_or_leaf_idx)
            # --------------------------------------------------------------
            orig_to_info = {}
            local_split_count = 0
            local_leaf_count = 0
            for i, node in enumerate(nodes):
                if node["is_leaf"]:
                    orig_to_info[i] = (True, local_leaf_count)
                    local_leaf_count += 1
                else:
                    orig_to_info[i] = (False, local_split_count)
                    local_split_count += 1

            # Root is always orig_idx=0
            root_is_leaf, root_local = orig_to_info[0]
            if root_is_leaf:
                # Edge-case: single-node tree (just a leaf).
                # Represent as a trivial split that always goes left.
                tree_roots.append(global_split_offset)
                all_nodes_modes.append(0)  # BRANCH_LEQ
                all_nodes_featureids.append(0)
                all_nodes_splits.append(float("inf"))
                all_nodes_truenodeids.append(global_leaf_offset)
                all_nodes_falsenodeids.append(global_leaf_offset)
                all_nodes_trueleafs.append(1)
                all_nodes_falseleafs.append(1)
                all_nodes_hitrates.append(1.0)
                all_nodes_missing_value_tracks_true.append(0)
                all_leaf_weights.append(float(nodes[0]["value"]))
                all_leaf_targetids.append(target_id)
                global_split_offset += 1
                global_leaf_offset += 1
                continue

            tree_roots.append(global_split_offset + root_local)

            # Build reverse maps for iteration
            split_local_to_orig = {}
            leaf_local_to_orig = {}
            for orig_i, (is_leaf, local_idx) in orig_to_info.items():
                if is_leaf:
                    leaf_local_to_orig[local_idx] = orig_i
                else:
                    split_local_to_orig[local_idx] = orig_i

            # --------------------------------------------------------------
            # Pass 2: fill in split-node arrays (in local split-index order)
            # --------------------------------------------------------------
            for local_split_idx in range(local_split_count):
                orig_i = split_local_to_orig[local_split_idx]
                node = nodes[orig_i]

                feat_id = int(node["feature_idx"])
                missing_go_left = int(node["missing_go_to_left"])

                true_orig = int(node["left"])
                false_orig = int(node["right"])
                true_is_leaf, true_local = orig_to_info[true_orig]
                false_is_leaf, false_local = orig_to_info[false_orig]

                if node["is_categorical"]:
                    mode = 6  # BRANCH_MEMBER
                    split_val = 0.0
                    bitset_idx = int(node["bitset_idx"])
                    left_cats = get_left_categories_from_bitset(
                        raw_left_cat_bitsets[bitset_idx]
                    )
                    membership_values_list.append(left_cats)
                else:
                    mode = 0  # BRANCH_LEQ
                    if "threshold" in node.dtype.names:
                        split_val = float(node["threshold"])
                    else:
                        split_val = float(node["num_threshold"])

                all_nodes_modes.append(mode)
                all_nodes_featureids.append(feat_id)
                all_nodes_splits.append(split_val)

                if true_is_leaf:
                    all_nodes_truenodeids.append(global_leaf_offset + true_local)
                    all_nodes_trueleafs.append(1)
                else:
                    all_nodes_truenodeids.append(global_split_offset + true_local)
                    all_nodes_trueleafs.append(0)

                if false_is_leaf:
                    all_nodes_falsenodeids.append(global_leaf_offset + false_local)
                    all_nodes_falseleafs.append(1)
                else:
                    all_nodes_falsenodeids.append(global_split_offset + false_local)
                    all_nodes_falseleafs.append(0)

                all_nodes_hitrates.append(1.0)
                all_nodes_missing_value_tracks_true.append(missing_go_left)

            # --------------------------------------------------------------
            # Pass 3: fill in leaf arrays (in local leaf-index order)
            # --------------------------------------------------------------
            for local_leaf_idx in range(local_leaf_count):
                orig_i = leaf_local_to_orig[local_leaf_idx]
                node = nodes[orig_i]
                all_leaf_weights.append(float(node["value"]))  # tree_weight = 1.0
                all_leaf_targetids.append(target_id)

            global_split_offset += local_split_count
            global_leaf_offset += local_leaf_count

    # ----------------------------------------------------------------
    # Build membership_values array (NaN-separated, in node-index order)
    # ----------------------------------------------------------------
    if membership_values_list:
        mv_flat = []
        for i, cats in enumerate(membership_values_list):
            if i > 0:
                mv_flat.append(float("nan"))
            mv_flat.extend(cats)
        membership_values_arr = np.array(mv_flat, dtype=dtype)
    else:
        membership_values_arr = np.array([], dtype=dtype)

    # ----------------------------------------------------------------
    # Build baseline-prediction initializer
    # ----------------------------------------------------------------
    if hasattr(op, "_baseline_prediction"):
        baseline = op._baseline_prediction
        if not isinstance(baseline, np.ndarray):
            baseline = np.array([baseline])
        baseline = np.array(baseline, dtype=dtype).ravel()
        if len(baseline) < n_targets:
            baseline = np.zeros(n_targets, dtype=dtype)
    else:
        baseline = np.zeros(n_targets, dtype=dtype)

    # ----------------------------------------------------------------
    # Emit the TreeEnsemble node (ai.onnx.ml opset 5)
    # ----------------------------------------------------------------
    from onnx import numpy_helper as _nph

    raw_scores_name = scope.get_unique_variable_name("hgb_raw_scores")

    container.add_node(
        "TreeEnsemble",
        [preprocessed_name],
        [raw_scores_name],
        op_domain="ai.onnx.ml",
        op_version=5,
        name=scope.get_unique_operator_name("TreeEnsemble"),
        n_targets=n_targets,
        aggregate_function=1,  # SUM
        post_transform=0,  # NONE
        tree_roots=tree_roots,
        nodes_modes=_nph.from_array(np.array(all_nodes_modes, dtype=np.uint8)),
        nodes_featureids=all_nodes_featureids,
        nodes_splits=_nph.from_array(np.array(all_nodes_splits, dtype=dtype)),
        nodes_truenodeids=all_nodes_truenodeids,
        nodes_falsenodeids=all_nodes_falsenodeids,
        nodes_trueleafs=all_nodes_trueleafs,
        nodes_falseleafs=all_nodes_falseleafs,
        nodes_hitrates=_nph.from_array(np.array(all_nodes_hitrates, dtype=dtype)),
        nodes_missing_value_tracks_true=all_nodes_missing_value_tracks_true,
        membership_values=_nph.from_array(membership_values_arr),
        leaf_weights=_nph.from_array(np.array(all_leaf_weights, dtype=dtype)),
        leaf_targetids=all_leaf_targetids,
    )

    # ----------------------------------------------------------------
    # Add baseline prediction: output = raw_scores + baseline
    # ----------------------------------------------------------------
    baseline_name = scope.get_unique_variable_name("hgb_baseline")
    container.add_initializer(
        baseline_name,
        (
            onnx_proto.TensorProto.FLOAT
            if dtype == np.float32
            else onnx_proto.TensorProto.DOUBLE
        ),
        baseline.shape,
        baseline.tolist(),
    )
    final_name = scope.get_unique_variable_name("hgb_final_scores")
    apply_add(scope, [raw_scores_name, baseline_name], final_name, container)
    return final_name


def _num_estimators(op):
    # don't use op.n_estimators since it may not be the same as
    # len(op.estimators_). At training time n_estimators can be changed by
    # training code:
    #   for j in range(10):
    #       ...
    #       classifier.fit(X_tmp, y_tmp)
    #       classifier.n_estimators += 30
    if hasattr(op, "estimators_"):
        return len(op.estimators_)
    elif hasattr(op, "_predictors"):
        # HistGradientBoosting*
        return len(op._predictors)
    raise NotImplementedError(
        "Model should have attribute 'estimators_' or '_predictors'."
    )


def _calculate_labels(scope, container, model, proba):
    predictions = []
    transposed_result_name = scope.get_unique_variable_name("transposed_result")
    apply_transpose(scope, proba, transposed_result_name, container, perm=(1, 2, 0))
    for k in range(model.n_outputs_):
        preds_name = scope.get_unique_variable_name("preds")
        reshaped_preds_name = scope.get_unique_variable_name("reshaped_preds")
        k_name = scope.get_unique_variable_name("k_column")
        out_k_name = scope.get_unique_variable_name("out_k_column")
        argmax_output_name = scope.get_unique_variable_name("argmax_output")
        classes_name = scope.get_unique_variable_name("classes")
        reshaped_result_name = scope.get_unique_variable_name("reshaped_result")

        container.add_initializer(k_name, onnx_proto.TensorProto.INT64, [], [k])
        container.add_initializer(
            classes_name,
            onnx_proto.TensorProto.INT64,
            model.classes_[k].shape,
            model.classes_[k],
        )

        container.add_node(
            "ArrayFeatureExtractor",
            [transposed_result_name, k_name],
            out_k_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        )
        container.add_node(
            "ArgMax",
            out_k_name,
            argmax_output_name,
            name=scope.get_unique_operator_name("ArgMax"),
            axis=1,
        )
        apply_reshape(
            scope,
            argmax_output_name,
            reshaped_result_name,
            container,
            desired_shape=(1, -1),
        )
        container.add_node(
            "ArrayFeatureExtractor",
            [classes_name, reshaped_result_name],
            preds_name,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
        )
        apply_reshape(
            scope, preds_name, reshaped_preds_name, container, desired_shape=(-1, 1)
        )
        predictions.append(reshaped_preds_name)
    return predictions


def convert_sklearn_random_forest_classifier(
    scope,
    operator,
    container,
    op_type="TreeEnsembleClassifier",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    attr_dtype = dtype if op_version >= 3 else np.float32
    op = operator.raw_operator

    # HistGradientBoosting with categorical features: use the new path.
    if hasattr(op, "_predictors") and _hgb_has_preprocessor(op):
        if hasattr(op, "n_trees_per_iteration_"):
            n_outputs = op.n_trees_per_iteration_
        else:
            raise NotImplementedError(
                "Model should have attribute 'n_trees_per_iteration_'."
            )
        options = container.get_options(op, dict(raw_scores=False))
        use_raw_scores = options["raw_scores"]

        preprocessed_name = _build_hgb_categorical_preproc(
            scope, container, operator, op, dtype
        )
        # n_targets = n_trees_per_iteration_ (1 for binary, n_classes for multi)
        final_scores_name = _build_hgb_tree_ensemble(
            scope, container, op, preprocessed_name, dtype, n_targets=n_outputs
        )

        # Determine the loss type to apply the correct post-transform
        if hasattr(op, "loss_"):
            loss = op.loss_
        elif hasattr(op, "_loss"):
            loss = op._loss
        else:
            loss = None

        classes = get_label_classes(scope, op)
        if all(isinstance(i, np.ndarray) for i in classes):
            classes = np.concatenate(classes)

        if use_raw_scores:
            # raw_scores: shape [N, n_outputs] - reshape to [N, 1] if binary
            if n_outputs == 1:
                apply_reshape(
                    scope,
                    final_scores_name,
                    operator.outputs[1].full_name,
                    container,
                    desired_shape=(-1, 1),
                )
            else:
                apply_reshape(
                    scope,
                    final_scores_name,
                    operator.outputs[1].full_name,
                    container,
                    desired_shape=(-1, n_outputs),
                )
        elif loss is not None and loss.__class__.__name__ in (
            "BinaryCrossEntropy",
            "HalfBinomialLoss",
        ):
            # Binary: apply sigmoid, build [1-p, p] probability matrix
            raw_flat = scope.get_unique_variable_name("hgb_cls_raw_flat")
            apply_reshape(
                scope, final_scores_name, raw_flat, container, desired_shape=(-1,)
            )
            prob1_name = scope.get_unique_variable_name("hgb_cls_prob1")
            container.add_node(
                "Sigmoid",
                [raw_flat],
                [prob1_name],
                op_version=13,
                name=scope.get_unique_operator_name("Sigmoid"),
            )
            # prob0 = 1 - prob1
            ones_name = scope.get_unique_variable_name("hgb_cls_ones")
            container.add_initializer(
                ones_name,
                (
                    onnx_proto.TensorProto.FLOAT
                    if dtype == np.float32
                    else onnx_proto.TensorProto.DOUBLE
                ),
                [],
                [1.0],
            )
            prob0_name = scope.get_unique_variable_name("hgb_cls_prob0")
            container.add_node(
                "Sub",
                [ones_name, prob1_name],
                [prob0_name],
                op_version=14,
                name=scope.get_unique_operator_name("Sub"),
            )
            # Reshape each to [N, 1] and concat -> [N, 2]
            p0_col = scope.get_unique_variable_name("hgb_cls_p0_col")
            p1_col = scope.get_unique_variable_name("hgb_cls_p1_col")
            apply_reshape(scope, prob0_name, p0_col, container, desired_shape=(-1, 1))
            apply_reshape(scope, prob1_name, p1_col, container, desired_shape=(-1, 1))
            apply_concat(
                scope,
                [p0_col, p1_col],
                operator.outputs[1].full_name,
                container,
                axis=1,
            )
        elif loss is not None and loss.__class__.__name__ in (
            "CategoricalCrossEntropy",
            "HalfMultinomialLoss",
        ):
            # Multi-class: apply softmax
            scores_2d = scope.get_unique_variable_name("hgb_cls_scores_2d")
            apply_reshape(
                scope,
                final_scores_name,
                scores_2d,
                container,
                desired_shape=(-1, n_outputs),
            )
            container.add_node(
                "Softmax",
                [scores_2d],
                [operator.outputs[1].full_name],
                op_version=13,
                name=scope.get_unique_operator_name("Softmax"),
                axis=1,
            )
        else:
            raise NotImplementedError(
                "Unsupported loss '{}' for HGB classifier with categorical features.".format(
                    loss.__class__.__name__ if loss is not None else "None"
                )
            )

        # Build class-label output (argmax of probabilities)
        if not use_raw_scores:
            argmax_name = scope.get_unique_variable_name("hgb_cls_argmax")
            container.add_node(
                "ArgMax",
                [operator.outputs[1].full_name],
                [argmax_name],
                op_version=13,
                name=scope.get_unique_operator_name("ArgMax"),
                axis=1,
                keepdims=0,
            )
            # Map argmax indices to actual class labels
            classes_name = scope.get_unique_variable_name("hgb_cls_classes")
            if all(isinstance(c, (numbers.Real, bool, np.bool_)) for c in classes):
                class_array = np.array([int(c) for c in classes], dtype=np.int64)
                container.add_initializer(
                    classes_name,
                    onnx_proto.TensorProto.INT64,
                    class_array.shape,
                    class_array.tolist(),
                )
                predicted_label = scope.get_unique_variable_name("hgb_cls_label")
                container.add_node(
                    "Gather",
                    [classes_name, argmax_name],
                    [predicted_label],
                    op_version=13,
                    name=scope.get_unique_operator_name("Gather"),
                    axis=0,
                )
            else:
                class_array = np.array([str(c) for c in classes])
                container.add_initializer(
                    classes_name,
                    onnx_proto.TensorProto.STRING,
                    class_array.shape,
                    class_array.tolist(),
                )
                predicted_label = scope.get_unique_variable_name("hgb_cls_label")
                container.add_node(
                    "Gather",
                    [classes_name, argmax_name],
                    [predicted_label],
                    op_version=13,
                    name=scope.get_unique_operator_name("Gather"),
                    axis=0,
                )
            apply_reshape(
                scope,
                predicted_label,
                operator.outputs[0].full_name,
                container,
                desired_shape=(-1,),
            )
        return

    if hasattr(op, "n_outputs_"):
        n_outputs = int(op.n_outputs_)
        options = container.get_options(
            op, dict(raw_scores=False, decision_path=False, decision_leaf=False)
        )
    elif hasattr(op, "n_trees_per_iteration_"):
        # HistGradientBoostingClassifier
        n_outputs = op.n_trees_per_iteration_
        options = container.get_options(op, dict(raw_scores=False))
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or 'n_trees_per_iteration_'."
        )

    use_raw_scores = options["raw_scores"]

    if n_outputs == 1 or hasattr(op, "loss_") or hasattr(op, "_loss"):
        classes = get_label_classes(scope, op)

        if all(isinstance(i, np.ndarray) for i in classes):
            classes = np.concatenate(classes)
        attr_pairs = get_default_tree_classifier_attribute_pairs()
        attr_pairs["name"] = scope.get_unique_operator_name(op_type)

        if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
            class_labels = [int(i) for i in classes]
            attr_pairs["classlabels_int64s"] = class_labels
        elif all(isinstance(i, str) for i in classes):
            class_labels = [str(i) for i in classes]
            attr_pairs["classlabels_strings"] = class_labels
        else:
            raise ValueError("Only string and integer class labels are allowed.")

        # random forest calculate the final score by averaging over all trees'
        # outcomes, so all trees' weights are identical.
        if hasattr(op, "estimators_"):
            estimator_count = len(op.estimators_)
            tree_weight = 1.0 / estimator_count
        elif hasattr(op, "_predictors"):
            # HistGradientBoostingRegressor
            estimator_count = len(op._predictors)
            tree_weight = 1.0
        else:
            raise NotImplementedError(
                "Model should have attribute 'estimators_' or '_predictors'."
            )

        for tree_id in range(estimator_count):
            if hasattr(op, "estimators_"):
                tree = op.estimators_[tree_id].tree_
                add_tree_to_attribute_pairs(
                    attr_pairs,
                    True,
                    tree,
                    tree_id,
                    tree_weight,
                    0,
                    True,
                    True,
                    dtype=dtype,
                )
            else:
                # HistGradientBoostClassifier
                if len(op._predictors[tree_id]) == 1:
                    tree = op._predictors[tree_id][0]
                    add_tree_to_attribute_pairs_hist_gradient_boosting(
                        attr_pairs,
                        True,
                        tree,
                        tree_id,
                        tree_weight,
                        0,
                        False,
                        False,
                        dtype=dtype,
                    )
                else:
                    for cl, tree in enumerate(op._predictors[tree_id]):
                        add_tree_to_attribute_pairs_hist_gradient_boosting(
                            attr_pairs,
                            True,
                            tree,
                            tree_id * n_outputs + cl,
                            tree_weight,
                            cl,
                            False,
                            False,
                            dtype=dtype,
                        )

        if hasattr(op, "_baseline_prediction"):
            if isinstance(op._baseline_prediction, np.ndarray):
                attr_pairs["base_values"] = list(op._baseline_prediction.ravel())
            else:
                attr_pairs["base_values"] = [op._baseline_prediction]

        if hasattr(op, "loss_"):
            loss = op.loss_
        elif hasattr(op, "_loss"):
            # scikit-learn >= 0.24
            loss = op._loss
        else:
            loss = None
        if loss is not None:
            if use_raw_scores:
                attr_pairs["post_transform"] = "NONE"
            elif loss.__class__.__name__ in ("BinaryCrossEntropy", "HalfBinomialLoss"):
                attr_pairs["post_transform"] = "LOGISTIC"
            elif loss.__class__.__name__ in (
                "CategoricalCrossEntropy",
                "HalfMultinomialLoss",
            ):
                attr_pairs["post_transform"] = "SOFTMAX"
            else:
                raise NotImplementedError(
                    "There is no corresponding post_transform for "
                    "'{}'.".format(loss.__class__.__name__)
                )
        elif use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}' and loss '{}'.".format(type(op), loss)
            )

        input_name = operator.input_full_names
        if isinstance(operator.inputs[0].type, BooleanTensorType):
            cast_input_name = scope.get_unique_variable_name("cast_input")

            apply_cast(
                scope,
                input_name,
                cast_input_name,
                container,
                to=onnx_proto.TensorProto.FLOAT,
            )
            input_name = cast_input_name

        if dtype is not None:
            for k in attr_pairs:
                if k in (
                    "nodes_values",
                    "class_weights",
                    "target_weights",
                    "nodes_hitrates",
                    "base_values",
                ):
                    attr_pairs[k] = np.array(attr_pairs[k], dtype=attr_dtype).ravel()

        container.add_node(
            op_type,
            input_name,
            [operator.outputs[0].full_name, operator.outputs[1].full_name],
            op_domain=op_domain,
            op_version=op_version,
            **attr_pairs,
        )

        if not options.get("decision_path", False) and not options.get(
            "decision_leaf", False
        ):
            return

        # decision_path
        tree_paths = []
        tree_leaves = []
        for i, tree in enumerate(op.estimators_):
            attrs = get_default_tree_classifier_attribute_pairs()
            attrs["name"] = scope.get_unique_operator_name("%s_%d" % (op_type, i))
            attrs["n_targets"] = int(op.n_outputs_)
            add_tree_to_attribute_pairs(
                attrs, True, tree.tree_, 0, 1.0, 0, False, True, dtype=dtype
            )

            attrs["n_targets"] = 1
            attrs["post_transform"] = "NONE"
            attrs["target_ids"] = [0 for _ in attrs["class_ids"]]
            attrs["target_weights"] = [float(_) for _ in attrs["class_nodeids"]]
            attrs["target_nodeids"] = attrs["class_nodeids"]
            attrs["target_treeids"] = attrs["class_treeids"]
            rem = [k for k in attrs if k.startswith("class")]
            for k in rem:
                del attrs[k]

            if dtype is not None:
                for k in attrs:
                    if k in (
                        "nodes_values",
                        "class_weights",
                        "target_weights",
                        "nodes_hitrates",
                        "base_values",
                    ):
                        attrs[k] = np.array(attrs[k], dtype=attr_dtype).ravel()

            if options["decision_path"]:
                # decision_path
                tree_paths.append(
                    _append_decision_output(
                        input_name,
                        attrs,
                        _build_labels_path,
                        None,
                        scope,
                        operator,
                        container,
                        op_type=op_type,
                        op_domain=op_domain,
                        op_version=op_version,
                        regression=True,
                        overwrite_tree=tree.tree_,
                    )
                )
            if options["decision_leaf"]:
                # decision_path
                tree_leaves.append(
                    _append_decision_output(
                        input_name,
                        attrs,
                        _build_labels_leaf,
                        None,
                        scope,
                        operator,
                        container,
                        op_type=op_type,
                        op_domain=op_domain,
                        op_version=op_version,
                        regression=True,
                        cast_encode=True,
                    )
                )

        # merges everything
        n_out = 2
        if options["decision_path"]:
            apply_concat(
                scope,
                tree_paths,
                operator.outputs[n_out].full_name,
                container,
                axis=1,
                operator_name=scope.get_unique_operator_name("concat"),
            )
            n_out += 1

        if options["decision_leaf"]:
            # decision_path
            apply_concat(
                scope,
                tree_leaves,
                operator.outputs[n_out].full_name,
                container,
                axis=1,
                operator_name=scope.get_unique_operator_name("concat"),
            )
            n_out += 1

    else:
        if use_raw_scores:
            raise RuntimeError(
                "The converter cannot implement decision_function for "
                "'{}'.".format(type(op))
            )
        concatenated_proba_name = scope.get_unique_variable_name("concatenated_proba")
        proba = []
        for est in op.estimators_:
            reshaped_est_proba_name = scope.get_unique_variable_name(
                "reshaped_est_proba"
            )
            est_proba = predict(
                est,
                scope,
                operator,
                container,
                op_type,
                op_domain,
                op_version,
                is_ensemble=True,
            )
            apply_reshape(
                scope,
                est_proba,
                reshaped_est_proba_name,
                container,
                desired_shape=(1, n_outputs, -1, max([len(x) for x in op.classes_])),
            )
            proba.append(reshaped_est_proba_name)
        apply_concat(scope, proba, concatenated_proba_name, container, axis=0)
        if container.target_opset >= 18:
            axis_name = scope.get_unique_variable_name("axis")
            container.add_initializer(axis_name, onnx_proto.TensorProto.INT64, [1], [0])
            container.add_node(
                "ReduceMean",
                [concatenated_proba_name, axis_name],
                operator.outputs[1].full_name,
                name=scope.get_unique_operator_name("ReduceMean"),
                keepdims=0,
            )
        else:
            container.add_node(
                "ReduceMean",
                concatenated_proba_name,
                operator.outputs[1].full_name,
                name=scope.get_unique_operator_name("ReduceMean"),
                axes=[0],
                keepdims=0,
            )
        predictions = _calculate_labels(
            scope, container, op, operator.outputs[1].full_name
        )
        apply_concat(
            scope, predictions, operator.outputs[0].full_name, container, axis=1
        )

        if options.get("decision_path", False) or options.get("decision_leaf", False):
            raise RuntimeError(
                "Decision output for multi-outputs is not implemented yet."
            )


def convert_sklearn_random_forest_regressor_converter(
    scope,
    operator,
    container,
    op_type="TreeEnsembleRegressor",
    op_domain="ai.onnx.ml",
    op_version=1,
):
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    op = operator.raw_operator

    # HistGradientBoosting with categorical features requires a different path:
    # the model's internal ColumnTransformer preprocessor must be replicated in
    # ONNX, and categorical splits must use the BRANCH_MEMBER mode of the new
    # TreeEnsemble op (ai.onnx.ml opset 5).
    if hasattr(op, "_predictors") and _hgb_has_preprocessor(op):
        preprocessed_name = _build_hgb_categorical_preproc(
            scope, container, operator, op, dtype
        )
        final_name = _build_hgb_tree_ensemble(
            scope, container, op, preprocessed_name, dtype
        )
        # Reshape to [N, 1] to match the expected output shape
        apply_reshape(
            scope,
            final_name,
            operator.outputs[0].full_name,
            container,
            desired_shape=(-1, 1),
        )
        return

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs["name"] = scope.get_unique_operator_name(op_type)

    if hasattr(op, "n_outputs_"):
        attrs["n_targets"] = int(op.n_outputs_)
    elif hasattr(op, "n_trees_per_iteration_"):
        # HistGradientBoostingRegressor
        attrs["n_targets"] = op.n_trees_per_iteration_
    else:
        raise NotImplementedError(
            "Model should have attribute 'n_outputs_' or 'n_trees_per_iteration_'."
        )

    if hasattr(op, "estimators_"):
        estimator_count = len(op.estimators_)
        tree_weight = 1.0 / estimator_count
    elif hasattr(op, "_predictors"):
        # HistGradientBoostingRegressor
        estimator_count = len(op._predictors)
        tree_weight = 1.0
    else:
        raise NotImplementedError(
            "Model should have attribute 'estimators_' or '_predictors'."
        )

    # random forest calculate the final score by averaging over all trees'
    # outcomes, so all trees' weights are identical.
    for tree_id in range(estimator_count):
        if hasattr(op, "estimators_"):
            tree = op.estimators_[tree_id].tree_
            add_tree_to_attribute_pairs(
                attrs, False, tree, tree_id, tree_weight, 0, False, True, dtype=dtype
            )
        else:
            # HistGradientBoostingRegressor
            if len(op._predictors[tree_id]) != 1:
                raise NotImplementedError(
                    "The converter does not work when the number of trees "
                    "is not 1 but {}.".format(len(op._predictors[tree_id]))
                )
            tree = op._predictors[tree_id][0]
            add_tree_to_attribute_pairs_hist_gradient_boosting(
                attrs, False, tree, tree_id, tree_weight, 0, False, False, dtype=dtype
            )

    if hasattr(op, "_baseline_prediction"):
        if isinstance(op._baseline_prediction, np.ndarray):
            attrs["base_values"] = list(op._baseline_prediction)
        else:
            attrs["base_values"] = [op._baseline_prediction]

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")

        apply_cast(
            scope,
            operator.input_full_names,
            cast_input_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        input_name = cast_input_name

    if dtype is not None:
        for k in attrs:
            if k in (
                "nodes_values",
                "class_weights",
                "target_weights",
                "nodes_hitrates",
                "base_values",
            ):
                attrs[k] = np.array(attrs[k], dtype=dtype).ravel()

    container.add_node(
        op_type,
        input_name,
        operator.outputs[0].full_name,
        op_domain=op_domain,
        op_version=op_version,
        **attrs,
    )

    if hasattr(op, "n_trees_per_iteration_"):
        # HistGradientBoostingRegressor does not implement decision_path.
        return
    if isinstance(op, RandomTreesEmbedding):
        options = scope.get_options(op)
    else:
        options = scope.get_options(op, dict(decision_path=False, decision_leaf=False))

    if not options.get("decision_path", False) and not options.get(
        "decision_leaf", False
    ):
        return

    # decision_path
    tree_paths = []
    tree_leaves = []
    for i, tree in enumerate(op.estimators_):
        attrs = get_default_tree_regressor_attribute_pairs()
        attrs["name"] = scope.get_unique_operator_name("%s_%d" % (op_type, i))
        attrs["n_targets"] = int(op.n_outputs_)
        add_tree_to_attribute_pairs(
            attrs, False, tree.tree_, 0, 1.0, 0, False, True, dtype=dtype
        )

        attrs["n_targets"] = 1
        attrs["post_transform"] = "NONE"
        attrs["target_ids"] = [0 for _ in attrs["target_ids"]]
        attrs["target_weights"] = [float(_) for _ in attrs["target_nodeids"]]

        if dtype is not None:
            for k in attrs:
                if k in (
                    "nodes_values",
                    "class_weights",
                    "target_weights",
                    "nodes_hitrates",
                    "base_values",
                ):
                    attrs[k] = np.array(attrs[k], dtype=dtype).ravel()

        if options.get("decision_path", False):
            # decision_path
            tree_paths.append(
                _append_decision_output(
                    input_name,
                    attrs,
                    _build_labels_path,
                    None,
                    scope,
                    operator,
                    container,
                    op_type=op_type,
                    op_domain=op_domain,
                    op_version=op_version,
                    regression=True,
                    overwrite_tree=tree.tree_,
                )
            )
        if options.get("decision_leaf", False):
            # decision_path
            tree_leaves.append(
                _append_decision_output(
                    input_name,
                    attrs,
                    _build_labels_leaf,
                    None,
                    scope,
                    operator,
                    container,
                    op_type=op_type,
                    op_domain=op_domain,
                    op_version=op_version,
                    regression=True,
                    cast_encode=True,
                )
            )

    # merges everything
    n_out = 1
    if options.get("decision_path", False):
        apply_concat(
            scope,
            tree_paths,
            operator.outputs[n_out].full_name,
            container,
            axis=1,
            operator_name=scope.get_unique_operator_name("concat"),
        )
        n_out += 1

    if options.get("decision_leaf", False):
        # decision_path
        apply_concat(
            scope,
            tree_leaves,
            operator.outputs[n_out].full_name,
            container,
            axis=1,
            operator_name=scope.get_unique_operator_name("concat"),
        )
        n_out += 1


register_converter(
    "SklearnRandomForestClassifier",
    convert_sklearn_random_forest_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "raw_scores": [True, False],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "decision_path": [True, False],
        "decision_leaf": [True, False],
    },
)
register_converter(
    "SklearnRandomForestRegressor",
    convert_sklearn_random_forest_regressor_converter,
    options={"decision_path": [True, False], "decision_leaf": [True, False]},
)
register_converter(
    "SklearnExtraTreesClassifier",
    convert_sklearn_random_forest_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "raw_scores": [True, False],
        "nocl": [True, False],
        "output_class_labels": [False, True],
        "decision_path": [True, False],
        "decision_leaf": [True, False],
    },
)
register_converter(
    "SklearnExtraTreesRegressor",
    convert_sklearn_random_forest_regressor_converter,
    options={"decision_path": [True, False], "decision_leaf": [True, False]},
)
register_converter(
    "SklearnHistGradientBoostingClassifier",
    convert_sklearn_random_forest_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "raw_scores": [True, False],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
register_converter(
    "SklearnHistGradientBoostingRegressor",
    convert_sklearn_random_forest_regressor_converter,
    options={"raw_scores": [True, False]},
)
