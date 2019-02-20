# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_concat
from ..common._registration import register_converter


def convert_sklearn_k_bins_discretiser(scope, operator, container):
    op = operator.raw_operator

    if op.encode == 'ohehot':
        raise RuntimeError("onehot encoding not supported since"
                           "ONNX does not support sparse tensors.")

    ranges = list(map(lambda e: e[1:-1], op.bin_edges_))
    digitised_output_name = [None] * len(ranges)

    for i in range(len(ranges)):
        digitised_output_name[i] = (
            scope.get_unique_variable_name('digitised_output_{}'.format(i)))
        column_index_name = scope.get_unique_variable_name('column_index')
        range_column_name = scope.get_unique_variable_name('range_column')
        column_name = scope.get_unique_variable_name('column')

        container.add_initializer(column_index_name,
                                  onnx_proto.TensorProto.INT64, [], [i])
        container.add_initializer(range_column_name,
                    onnx_proto.TensorProto.FLOAT, [len(ranges[i])], ranges[i])

        container.add_node('ArrayFeatureExtractor',
                [operator.inputs[0].full_name, column_index_name], column_name,
                name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                op_domain='ai.onnx.ml')
        # Digitize op needs to be added to onnx
        # https://aiinfra.visualstudio.com/Lotus/_workitems/edit/2619
        container.add_node('Digitize', [column_name, range_column_name],
                           digitised_output_name[i])

    if op.encode == 'onehot-dense':
        concat_output_name = scope.get_unique_variable_name('concat_output')

        apply_concat(scope, [d for d in digitised_output_name],
                     concat_output_name, container, axis=0)
        container.add_node('OneHotEncoder', concat_output_name,
                operator.output_full_names,
                name=scope.get_unique_operator_name('OneHotEncoder'),
                cats_int64s=list(map(lambda x: len(range(x)), op.n_bins_)),
                op_domain='ai.onnx.ml')
    else:
        apply_concat(scope, [d for d in digitised_output_name],
                     operator.output_full_names, container, axis=0)


register_converter('SklearnKBinsDiscretizer',
                   convert_sklearn_k_bins_discretiser)
