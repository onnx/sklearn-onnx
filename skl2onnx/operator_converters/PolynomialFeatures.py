# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..proto import onnx_proto
from ..common._apply_operation import apply_concat, apply_cast
from ..common._registration import register_converter


def convert_sklearn_polynomial_features(scope, operator, container):
    op = operator.raw_operator
    transformed_columns = [None] * (op.n_output_features_)

    combinations = op._combinations(op.n_input_features_, op.degree,
                                    op.interaction_only,
                                    op.include_bias)

    for i, comb in enumerate(combinations):
        if len(comb) == 0:
            unit_name = scope.get_unique_variable_name('unit')
            array = np.ones((operator.inputs[0].type.shape[0], 1))

            container.add_initializer(unit_name, onnx_proto.TensorProto.FLOAT,
                                      array.shape, array.flatten())

            transformed_columns[i] = unit_name
        else:
            comb_name = scope.get_unique_variable_name('comb')
            col_name = scope.get_unique_variable_name('col')
            prod_name = scope.get_unique_variable_name('prod')

            container.add_initializer(comb_name, onnx_proto.TensorProto.INT64,
                                      [len(comb)], list(comb))

            container.add_node(
                'ArrayFeatureExtractor',
                [operator.inputs[0].full_name, comb_name], col_name,
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
            reduce_prod_input = col_name
            if (operator.inputs[0].type._get_element_onnx_type()
                    == onnx_proto.TensorProto.INT64):
                float_col_name = scope.get_unique_variable_name('col')

                apply_cast(scope, col_name, float_col_name, container,
                           to=onnx_proto.TensorProto.FLOAT)
                reduce_prod_input = float_col_name

            container.add_node(
                'ReduceProd', reduce_prod_input, prod_name,
                axes=[1], name=scope.get_unique_operator_name('ReduceProd'))
            transformed_columns[i] = prod_name

    if (operator.inputs[0].type._get_element_onnx_type()
            == onnx_proto.TensorProto.INT64):
        concat_result_name = scope.get_unique_variable_name('concat_result')

        apply_concat(scope, [t for t in transformed_columns],
                     concat_result_name, container, axis=1)
        apply_cast(scope, concat_result_name,
                   operator.outputs[0].full_name, container,
                   to=onnx_proto.TensorProto.INT64)
    else:
        apply_concat(scope, [t for t in transformed_columns],
                     operator.outputs[0].full_name, container, axis=1)


register_converter('SklearnPolynomialFeatures',
                   convert_sklearn_polynomial_features)
