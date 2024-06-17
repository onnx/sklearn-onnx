# SPDX-License-Identifier: Apache-2.0

from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import Int64TensorType
from .._supported_operators import sklearn_operator_name_map


def convert_sklearn_tuned_threshold_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    estimator = operator.raw_operator.estimator_
    op_type = sklearn_operator_name_map[type(estimator)]

    this_operator = scope.declare_local_operator(op_type, estimator)
    this_operator.inputs = operator.inputs

    label_name = scope.declare_local_variable("label_tuned", Int64TensorType())
    prob_name = scope.declare_local_variable(
        "proba_tuned", operator.outputs[1].type.__class__()
    )
    this_operator.outputs.append(label_name)
    this_operator.outputs.append(prob_name)

    container.add_node(
        "Identity", [label_name.onnx_name], [operator.outputs[0].full_name]
    )
    container.add_node(
        "Identity", [prob_name.onnx_name], [operator.outputs[1].full_name]
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
