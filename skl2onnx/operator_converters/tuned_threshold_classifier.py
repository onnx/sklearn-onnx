# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import Int64TensorType
from ..common._apply_operation import apply_cast, apply_reshape
from .._supported_operators import sklearn_operator_name_map
from ..proto import onnx_proto


def _convert_threshold_classifier(
    scope: Scope,
    operator: Operator,
    container: ModelComponentContainer,
    threshold: float,
    pos_label_idx: int,
    neg_label_idx: int,
):
    """
    Shared conversion logic for classifiers that apply a custom decision
    threshold to the inner estimator's probabilities.

    Parameters
    ----------
    threshold : float
        The decision threshold to apply to the positive class probability.
    pos_label_idx : int
        Index of the positive class in ``classes_``.
    neg_label_idx : int
        Index of the negative class in ``classes_``.
    """
    op = operator.raw_operator
    estimator = op.estimator_
    op_type = sklearn_operator_name_map[type(estimator)]

    this_operator = scope.declare_local_operator(op_type, estimator)
    this_operator.inputs = operator.inputs

    label_name = scope.declare_local_variable("label_tuned", Int64TensorType())
    prob_name = scope.declare_local_variable(
        "proba_tuned", operator.outputs[1].type.__class__()
    )
    this_operator.outputs.append(label_name)
    this_operator.outputs.append(prob_name)

    # Pass through the probabilities unchanged
    container.add_node(
        "Identity", [prob_name.onnx_name], [operator.outputs[1].full_name]
    )

    # Extract probability of the positive class: proba[:, pos_label_idx]
    pos_col_name = scope.get_unique_variable_name("pos_col_idx")
    container.add_initializer(
        pos_col_name, onnx_proto.TensorProto.INT64, [1], [pos_label_idx]
    )
    pos_proba_name = scope.get_unique_variable_name("pos_proba")
    container.add_node(
        "Gather",
        [prob_name.onnx_name, pos_col_name],
        pos_proba_name,
        name=scope.get_unique_operator_name("Gather"),
        axis=1,
    )
    pos_proba_flat_name = scope.get_unique_variable_name("pos_proba_flat")
    apply_reshape(
        scope,
        pos_proba_name,
        pos_proba_flat_name,
        container,
        desired_shape=(-1,),
    )

    # Compare with threshold: is_positive = (pos_proba >= threshold).
    # Use float32 for the threshold since classifier probabilities are always
    # output as float32 regardless of the declared output type.
    threshold_name = scope.get_unique_variable_name("threshold")
    container.add_initializer(
        threshold_name, onnx_proto.TensorProto.FLOAT, [], [float(threshold)]
    )
    is_pos_name = scope.get_unique_variable_name("is_positive")
    container.add_node(
        "GreaterOrEqual",
        [pos_proba_flat_name, threshold_name],
        is_pos_name,
        name=scope.get_unique_operator_name("GreaterOrEqual"),
    )

    # Cast bool to int64 (0 or 1) to index into ordered classes array
    is_pos_int_name = scope.get_unique_variable_name("is_positive_int")
    apply_cast(
        scope,
        is_pos_name,
        is_pos_int_name,
        container,
        to=onnx_proto.TensorProto.INT64,
    )

    # Build ordered classes array: [classes[neg_label_idx], classes[pos_label_idx]]
    classes = op.classes_
    if np.issubdtype(classes.dtype, np.floating) or classes.dtype == np.bool_:
        class_type = onnx_proto.TensorProto.INT32
        ordered_classes = np.array(
            [int(classes[neg_label_idx]), int(classes[pos_label_idx])],
            dtype=np.int32,
        )
    elif np.issubdtype(classes.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
        ordered_classes = np.array(
            [int(classes[neg_label_idx]), int(classes[pos_label_idx])],
            dtype=np.int32,
        )
    else:
        class_type = onnx_proto.TensorProto.STRING
        ordered_classes = np.array(
            [
                classes[neg_label_idx].encode("utf-8"),
                classes[pos_label_idx].encode("utf-8"),
            ]
        )

    ordered_classes_name = scope.get_unique_variable_name("ordered_classes")
    container.add_initializer(
        ordered_classes_name, class_type, [2], ordered_classes.tolist()
    )

    # Gather labels: ordered_classes[is_pos_int]
    gathered_name = scope.get_unique_variable_name("gathered_label")
    container.add_node(
        "Gather",
        [ordered_classes_name, is_pos_int_name],
        gathered_name,
        name=scope.get_unique_operator_name("Gather"),
        axis=0,
    )

    if class_type == onnx_proto.TensorProto.STRING:
        container.add_node(
            "Identity",
            [gathered_name],
            [operator.outputs[0].full_name],
        )
    else:
        apply_cast(
            scope,
            gathered_name,
            operator.outputs[0].full_name,
            container,
            to=onnx_proto.TensorProto.INT64,
        )


def convert_sklearn_tuned_threshold_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    classes = op.classes_
    pos_label = op._curve_scorer._get_pos_label()
    if pos_label is None:
        pos_label_idx = 1
        neg_label_idx = 0
    else:
        pos_label_idx = int(np.flatnonzero(classes == pos_label)[0])
        neg_label_idx = int(np.flatnonzero(classes != pos_label)[0])

    _convert_threshold_classifier(
        scope,
        operator,
        container,
        threshold=op.best_threshold_,
        pos_label_idx=pos_label_idx,
        neg_label_idx=neg_label_idx,
    )


def convert_sklearn_fixed_threshold_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    classes = op.classes_
    pos_label = op.pos_label
    if pos_label is None:
        pos_label_idx = 1
        neg_label_idx = 0
    else:
        pos_label_idx = int(np.flatnonzero(classes == pos_label)[0])
        neg_label_idx = int(np.flatnonzero(classes != pos_label)[0])

    threshold = op.threshold
    if threshold == "auto":
        # When threshold is "auto", use 0.5 for predict_proba (the default
        # response method when available).
        threshold = 0.5

    _convert_threshold_classifier(
        scope,
        operator,
        container,
        threshold=threshold,
        pos_label_idx=pos_label_idx,
        neg_label_idx=neg_label_idx,
    )


register_converter(
    "SklearnTunedThresholdClassifierCV",
    convert_sklearn_tuned_threshold_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)

register_converter(
    "SklearnFixedThresholdClassifier",
    convert_sklearn_fixed_threshold_classifier,
    options={
        "zipmap": [True, False, "columns"],
        "output_class_labels": [False, True],
        "nocl": [True, False],
    },
)
