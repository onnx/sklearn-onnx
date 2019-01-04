# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._registration import register_converter


def convert_sklearn_array_feature_extractor(scope, operator, container):
    column_indices_name = scope.get_unique_variable_name('column_indices')

    container.add_initializer(column_indices_name, onnx_proto.TensorProto.INT64,
                              [len(operator.column_indices)], operator.column_indices)

    container.add_node('ArrayFeatureExtractor', [operator.inputs[0].full_name, column_indices_name],
                       operator.outputs[0].full_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                       op_domain='ai.onnx.ml')


register_converter('SklearnArrayFeatureExtractor', convert_sklearn_array_feature_extractor)
