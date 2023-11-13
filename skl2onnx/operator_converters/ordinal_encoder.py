# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..common._apply_operation import apply_cast, apply_concat, apply_reshape
from ..common._container import ModelComponentContainer
from ..common.data_types import DoubleTensorType, Int32TensorType, \
    Int16TensorType
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..proto import onnx_proto


def convert_sklearn_ordinal_encoder(
        scope: Scope, operator: Operator, container: ModelComponentContainer
):
    ordinal_op = operator.raw_operator
    result = []
    input_idx = 0
    dimension_idx = 0
    for categories in ordinal_op.categories_:
        if len(categories) == 0:
            continue

        current_input = operator.inputs[input_idx]
        if current_input.get_second_dimension() == 1:
            feature_column_name = current_input.onnx_name
            input_idx += 1
        else:
            index_name = scope.get_unique_variable_name("index")
            container.add_initializer(
                index_name, onnx_proto.TensorProto.INT64, [], [dimension_idx]
            )

            feature_column_name = scope.get_unique_variable_name(
                "feature_column")

            container.add_node(
                "ArrayFeatureExtractor",
                [current_input.onnx_name, index_name],
                feature_column_name,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
            )

            dimension_idx += 1
            if dimension_idx == current_input.get_second_dimension():
                dimension_idx = 0
                input_idx += 1

        to = None
        if isinstance(current_input.type, DoubleTensorType):
            to = onnx_proto.TensorProto.FLOAT
        if isinstance(current_input.type, (Int16TensorType, Int32TensorType)):
            to = onnx_proto.TensorProto.INT64
        if to is not None:
            casted_feature_column_name = scope.get_unique_variable_name(
                'casted_feature_column')

            apply_cast(
                scope, feature_column_name, casted_feature_column_name,
                container, to=to)

            feature_column_name = casted_feature_column_name

        attrs = {"name": scope.get_unique_operator_name("LabelEncoder")}
        if (
                np.issubdtype(categories.dtype, np.floating)
                or categories.dtype == np.bool_
                or isinstance(categories[0], float)
        ):
            attrs["keys_floats"] = categories
        elif (
                np.issubdtype(categories.dtype, np.signedinteger)
                or isinstance(categories[0], int)
        ):
            attrs["keys_int64s"] = categories
        else:
            attrs["keys_strings"] = np.array(
                [str(s).encode("utf-8") for s in categories]
            )
        attrs["values_int64s"] = np.arange(len(categories)).astype(np.int64)

        result.append(scope.get_unique_variable_name("ordinal_output"))
        label_encoder_output = scope.get_unique_variable_name(
            "label_encoder")

        container.add_node(
            "LabelEncoder",
            feature_column_name,
            label_encoder_output,
            op_domain="ai.onnx.ml",
            op_version=2,
            **attrs
        )
        apply_reshape(
            scope,
            label_encoder_output,
            result[-1],
            container,
            desired_shape=(-1, 1),
        )

    concat_result_name = scope.get_unique_variable_name("concat_result")
    apply_concat(scope, result, concat_result_name, container, axis=1)
    cast_type = (
        onnx_proto.TensorProto.FLOAT
        if np.issubdtype(ordinal_op.dtype, np.floating)
        else onnx_proto.TensorProto.INT64
    )
    apply_cast(
        scope, concat_result_name, operator.output_full_names, container,
        to=cast_type
    )


register_converter("SklearnOrdinalEncoder", convert_sklearn_ordinal_encoder)
