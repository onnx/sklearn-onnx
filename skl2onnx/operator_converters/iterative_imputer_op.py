# SPDX-License-Identifier: Apache-2.0


import numpy as np
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..common.data_types import (
    FloatTensorType,
    DoubleTensorType,
    guess_proto_type,
)
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from .._supported_operators import sklearn_operator_name_map
from .._parse import _parse_sklearn


def _make_tensor_type(dtype, shape):
    """Return FloatTensorType or DoubleTensorType for the given numpy dtype."""
    if dtype == np.float64:
        return DoubleTensorType(shape)
    return FloatTensorType(shape)


def convert_sklearn_iterative_imputer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts *IterativeImputer* into *ONNX*.

    The conversion unrolls the fitted estimators across all imputation
    rounds and features, building an ONNX graph that replicates
    ``IterativeImputer.transform`` exactly.
    """
    op = operator.raw_operator
    if not hasattr(op, "initial_imputer_"):
        raise RuntimeError(
            "Member initial_imputer_ is not present, was the model fitted?"
        )

    proto_type = guess_proto_type(operator.inputs[0].type)
    if proto_type != onnx_proto.TensorProto.DOUBLE:
        proto_type = onnx_proto.TensorProto.FLOAT
    dtype = np.float64 if proto_type == onnx_proto.TensorProto.DOUBLE else np.float32

    n_features = op.n_features_in_
    X_orig_name = operator.inputs[0].full_name

    # Compute missing mask from original input (before imputation).
    # The mask is needed throughout the unrolled loop to determine which
    # positions to update with predicted values vs. keeping the original.
    is_nan_name = scope.get_unique_variable_name("is_nan")
    missing_values = op.missing_values
    if isinstance(missing_values, float) and np.isnan(missing_values):
        container.add_node(
            "IsNaN",
            [X_orig_name],
            [is_nan_name],
            name=scope.get_unique_operator_name("IsNaN"),
        )
    else:
        mv_name = scope.get_unique_variable_name("missing_val")
        container.add_initializer(
            mv_name, proto_type, [], [dtype(missing_values)]
        )
        container.add_node(
            "Equal",
            [X_orig_name, mv_name],
            [is_nan_name],
            name=scope.get_unique_operator_name("Equal_nan"),
        )

    # Step 1: Initial imputation using initial_imputer_ (a SimpleImputer).
    # _parse_sklearn declares the sub-operator which will be converted later.
    init_imp_outputs = _parse_sklearn(
        scope, op.initial_imputer_, operator.inputs
    )
    current_x_name = init_imp_outputs[0].full_name

    # Precompute per-feature constants that are reused across iterations.
    # Each unique feat_idx needs: a [1]-shaped column-index initializer,
    # an update_mask [1, F] with 1 at feat_idx, and a keep_mask [1, F]
    # with 0 at feat_idx.  These are constant and shared across all steps
    # that target the same feature.
    unique_feat_indices = {int(t.feat_idx) for t in op.imputation_sequence_}
    col_idx_names = {}   # feat_idx -> initializer name holding [feat_idx]
    upd_mask_names = {}  # feat_idx -> initializer name for update mask
    kp_mask_names = {}   # feat_idx -> initializer name for keep mask

    for fi in sorted(unique_feat_indices):
        col_idx_name = scope.get_unique_variable_name("col_idx_f%d" % fi)
        container.add_initializer(
            col_idx_name,
            onnx_proto.TensorProto.INT64,
            [1],
            [fi],
        )
        col_idx_names[fi] = col_idx_name

        update_mask = np.zeros((1, n_features), dtype=dtype)
        update_mask[0, fi] = dtype(1)
        keep_mask = np.ones((1, n_features), dtype=dtype) - update_mask

        upd_mask_name = scope.get_unique_variable_name("upd_mask_f%d" % fi)
        container.add_initializer(
            upd_mask_name,
            proto_type,
            [1, n_features],
            update_mask.flatten().tolist(),
        )
        upd_mask_names[fi] = upd_mask_name

        kp_mask_name = scope.get_unique_variable_name("kp_mask_f%d" % fi)
        container.add_initializer(
            kp_mask_name,
            proto_type,
            [1, n_features],
            keep_mask.flatten().tolist(),
        )
        kp_mask_names[fi] = kp_mask_name

    # Step 2: Unroll all (iteration × feature) steps from imputation_sequence_.
    # sklearn stores the full sequence for all iterations so we iterate once.
    for step_idx, triplet in enumerate(op.imputation_sequence_):
        feat_idx = int(triplet.feat_idx)
        neighbor_feat_idx = [int(i) for i in triplet.neighbor_feat_idx]
        estimator = triplet.estimator
        n_neighbors = len(neighbor_feat_idx)

        sfx = "_%d" % step_idx

        # --- Gather neighbor columns from current X ---
        nb_idx_name = scope.get_unique_variable_name("nb_idx" + sfx)
        container.add_initializer(
            nb_idx_name,
            onnx_proto.TensorProto.INT64,
            [n_neighbors],
            neighbor_feat_idx,
        )

        nb_x_var = scope.declare_local_variable(
            "nb_x" + sfx, _make_tensor_type(dtype, [None, n_neighbors])
        )
        container.add_node(
            "Gather",
            [current_x_name, nb_idx_name],
            [nb_x_var.full_name],
            name=scope.get_unique_operator_name("Gather_nb" + sfx),
            axis=1,
        )

        # --- Apply sub-estimator to get predictions ---
        try:
            est_op_type = sklearn_operator_name_map[type(estimator)]
        except KeyError:
            raise RuntimeError(
                "IterativeImputer uses estimator %r which is not supported "
                "by sklearn-onnx. You may raise an issue at "
                "https://github.com/onnx/sklearn-onnx/issues."
                % type(estimator).__name__
            )
        est_operator = scope.declare_local_operator(est_op_type, estimator)
        est_operator.inputs.append(nb_x_var)

        pred_var = scope.declare_local_variable(
            "pred" + sfx,
            operator.outputs[0].type.__class__(),
        )
        est_operator.outputs.append(pred_var)
        pred_name = pred_var.full_name

        # --- Gather current column value and NaN mask for feat_idx ---
        col_idx_name = col_idx_names[feat_idx]

        current_col_name = scope.get_unique_variable_name("cur_col" + sfx)
        container.add_node(
            "Gather",
            [current_x_name, col_idx_name],
            [current_col_name],
            name=scope.get_unique_operator_name("Gather_col" + sfx),
            axis=1,
        )
        # current_col_name has shape [N, 1]

        nan_col_name = scope.get_unique_variable_name("nan_col" + sfx)
        container.add_node(
            "Gather",
            [is_nan_name, col_idx_name],
            [nan_col_name],
            name=scope.get_unique_operator_name("Gather_nan" + sfx),
            axis=1,
        )
        # nan_col_name has shape [N, 1]

        # --- Where(is_nan_col, pred, current_col): use prediction for NaN ---
        new_col_name = scope.get_unique_variable_name("new_col" + sfx)
        container.add_node(
            "Where",
            [nan_col_name, pred_name, current_col_name],
            [new_col_name],
            name=scope.get_unique_operator_name("Where" + sfx),
        )
        # new_col_name has shape [N, 1]

        # --- Update X_current: replace column feat_idx with new_col ---
        # Strategy: X_new = X_current * keep_mask + new_col * update_mask
        # Broadcasting: new_col [N,1] * update_mask [1,F] -> [N,F]
        upd_mask_name = upd_mask_names[feat_idx]
        kp_mask_name = kp_mask_names[feat_idx]

        x_kept_name = scope.get_unique_variable_name("x_kept" + sfx)
        container.add_node(
            "Mul",
            [current_x_name, kp_mask_name],
            [x_kept_name],
            name=scope.get_unique_operator_name("Mul_kp" + sfx),
        )

        col_bc_name = scope.get_unique_variable_name("col_bc" + sfx)
        container.add_node(
            "Mul",
            [new_col_name, upd_mask_name],
            [col_bc_name],
            name=scope.get_unique_operator_name("Mul_bc" + sfx),
        )

        new_x_name = scope.get_unique_variable_name("new_x" + sfx)
        container.add_node(
            "Add",
            [x_kept_name, col_bc_name],
            [new_x_name],
            name=scope.get_unique_operator_name("Add_xnew" + sfx),
        )

        current_x_name = new_x_name

    # Connect final result to operator output
    container.add_node(
        "Identity",
        [current_x_name],
        [operator.outputs[0].full_name],
        name=scope.get_unique_operator_name("Id_out"),
    )


register_converter(
    "SklearnIterativeImputer", convert_sklearn_iterative_imputer
)
