# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..proto import onnx_proto
from ..common._apply_operation import apply_concat, apply_reshape
from ..common._registration import register_converter


def convert_sklearn_one_hot_encoder(scope, operator, container):
    op = operator.raw_operator
    op_type = 'OneHotEncoder'
    result = []
    concat_result_name = scope.get_unique_variable_name('concat_result')
    categories_len = 0

    for index, categories in enumerate(op.categories_):
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        attrs['zeros'] = 1 if op.handle_unknown == 'ignore' else 0
        if hasattr(op, 'drop_idx_') and op.drop_idx_ is not None:
            categories = (categories[np.arange(len(categories)) !=
                          op.drop_idx_[index]])
        if len(categories) > 0:
            if (np.issubdtype(categories.dtype, np.floating)
                    or np.issubdtype(categories.dtype, np.signedinteger)):
                attrs['cats_int64s'] = categories.astype(np.int64)
            else:
                attrs['cats_strings'] = np.array(
                    [s.encode('utf-8') for s in categories])

            index_name = scope.get_unique_variable_name('index')
            feature_column_name = scope.get_unique_variable_name(
                'feature_column')
            result.append(scope.get_unique_variable_name('ohe_output'))

            container.add_initializer(
                index_name, onnx_proto.TensorProto.INT64, [], [index])

            container.add_node(
                'ArrayFeatureExtractor',
                [operator.inputs[0].full_name, index_name],
                feature_column_name, op_domain='ai.onnx.ml',
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'))

            container.add_node('OneHotEncoder', feature_column_name,
                               result[-1], op_domain='ai.onnx.ml', **attrs)
            categories_len += len(categories)
    apply_concat(scope, result,
                 concat_result_name, container, axis=2)
    apply_reshape(scope, concat_result_name, operator.output_full_names,
                  container, desired_shape=(-1, categories_len))


register_converter('SklearnOneHotEncoder', convert_sklearn_one_hot_encoder)
