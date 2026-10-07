# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._apply_operation import apply_cast, apply_normalizer
from ..common.data_types import DoubleTensorType, FloatTensorType
from ..proto import onnx_proto
from .common import concatenate_variables


def convert_sklearn_normalizer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    if len(operator.inputs) > 1:
        # If there are multiple input tensors,
        # we combine them using a FeatureVectorizer
        feature_name = concatenate_variables(scope, operator.inputs, container)
    else:
        # No concatenation is needed, we just use the first variable's name
        feature_name = operator.inputs[0].full_name
    op = operator.raw_operator
    norm_map = {"max": "MAX", "l1": "L1", "l2": "L2"}
    if op.norm in norm_map:
        norm = norm_map[op.norm]
    else:
        raise RuntimeError(
            "Invalid norm '%s'. You may raise an issue"
            "at https://github.com/onnx/sklearn-onnx/"
            "issues." % op.norm
        )
    use_float = type(operator.inputs[0].type) not in (DoubleTensorType,)
    if norm == "MAX" and type(operator.inputs[0].type) not in (
        FloatTensorType,
        DoubleTensorType,
    ):
        # Norm MAX is computed with standard operators which keep
        # the integer type, the output is float.
        cast_name = scope.get_unique_variable_name("cast_input")
        apply_cast(
            scope, feature_name, cast_name, container, to=onnx_proto.TensorProto.FLOAT
        )
        feature_name = cast_name
    apply_normalizer(
        scope,
        feature_name,
        operator.outputs[0].full_name,
        container,
        norm=norm,
        use_float=use_float,
        dtype=np.float32 if use_float else np.float64,
    )


register_converter("SklearnNormalizer", convert_sklearn_normalizer)
