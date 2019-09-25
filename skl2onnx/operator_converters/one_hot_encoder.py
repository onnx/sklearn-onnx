# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._apply_operation import apply_cast, apply_concat, apply_reshape
from ..common.data_types import Int64TensorType, StringTensorType
from ..common._registration import register_converter
from ..proto import onnx_proto


def convert_sklearn_one_hot_encoder(scope, operator, container):
    ohe_op = operator.raw_operator
    result, categories_len = [], 0
    concatenated_input_name = operator.inputs[0].full_name
    concat_result_name = scope.get_unique_variable_name('concat_result')

    if len(operator.inputs) > 1:
        concatenated_input_name = scope.get_unique_variable_name(
            'concatenated_input')
        if all(isinstance(inp.type, type(operator.inputs[0].type))
               for inp in operator.inputs):
            input_names = list(map(lambda x: x.full_name, operator.inputs))
        else:
            input_names = []
            for inp in operator.inputs:
                if isinstance(inp.type, Int64TensorType):
                    input_names.append(scope.get_unique_variable_name(
                        'cast_input'))
                    apply_cast(scope, inp.full_name, input_names[-1],
                               container, to=onnx_proto.TensorProto.STRING)
                elif isinstance(inp.type, StringTensorType):
                    input_names.append(inp.full_name)
                else:
                    raise NotImplementedError(
                        "{} input datatype not yet supported. "
                        "You may raise an issue at "
                        "https://github.com/onnx/sklearn-onnx/issues"
                        "".format(type(inp.type)))

        apply_concat(scope, input_names,
                     concatenated_input_name, container, axis=1)
    for index, categories in enumerate(ohe_op.categories_):
        attrs = {'name': scope.get_unique_operator_name('OneHotEncoder')}
        attrs['zeros'] = 1 if ohe_op.handle_unknown == 'ignore' else 0
        if hasattr(ohe_op, 'drop_idx_') and ohe_op.drop_idx_ is not None:
            categories = (categories[np.arange(len(categories)) !=
                                     ohe_op.drop_idx_[index]])
        if len(categories) > 0:
            if (np.issubdtype(categories.dtype, np.floating)
                    or np.issubdtype(categories.dtype, np.signedinteger)):
                attrs['cats_int64s'] = categories.astype(np.int64)
            else:
                attrs['cats_strings'] = np.array(
                    [str(s).encode('utf-8') for s in categories])

            index_name = scope.get_unique_variable_name('index')
            feature_column_name = scope.get_unique_variable_name(
                'feature_column')
            result.append(scope.get_unique_variable_name('ohe_output'))

            container.add_initializer(
                index_name, onnx_proto.TensorProto.INT64, [], [index])

            container.add_node(
                'ArrayFeatureExtractor',
                [concatenated_input_name, index_name],
                feature_column_name, op_domain='ai.onnx.ml',
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'))

            container.add_node('OneHotEncoder', feature_column_name,
                               result[-1], op_domain='ai.onnx.ml', **attrs)
            categories_len += len(categories)
    apply_concat(scope, result,
                 concat_result_name, container, axis=2)
    reshape_input = concat_result_name
    if np.issubdtype(ohe_op.dtype, np.signedinteger):
        reshape_input = scope.get_unique_variable_name('cast')
        apply_cast(scope, concat_result_name, reshape_input,
                   container, to=onnx_proto.TensorProto.INT64)
    apply_reshape(scope, reshape_input, operator.output_full_names,
                  container, desired_shape=(-1, categories_len))


register_converter('SklearnOneHotEncoder', convert_sklearn_one_hot_encoder)
