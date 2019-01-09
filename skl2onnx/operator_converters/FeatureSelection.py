# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_cast
from ..common._registration import register_converter


def convert_sklearn_feature_selection(scope, operator, container):
    op = operator.raw_operator
    index = op.get_support(indices=True)  # Get indices of the features selected

    if index:
        column_indices_name = scope.get_unique_variable_name('column_indices')

        container.add_initializer(column_indices_name, onnx_proto.TensorProto.INT64,
                                  [len(index)], index)

        container.add_node('ArrayFeatureExtractor', [operator.inputs[0].full_name, column_indices_name],
                           operator.outputs[0].full_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                           op_domain='ai.onnx.ml')
    else:
        empty_output_name = scope.get_unique_variable_name('empty_output')

        container.add_node('ConstantOfShape', operator.inputs[0].full_name, empty_output_name,
                           op_version=9)
        apply_cast(scope, empty_output_name, operator.outputs[0].full_name, container,
                   to=onnx_proto.TensorProto.FLOAT)


register_converter('SklearnGenericUnivariateSelect', convert_sklearn_feature_selection)
register_converter('SklearnRFE', convert_sklearn_feature_selection)
register_converter('SklearnRFECV', convert_sklearn_feature_selection)
register_converter('SklearnSelectFdr', convert_sklearn_feature_selection)
register_converter('SklearnSelectFpr', convert_sklearn_feature_selection)
register_converter('SklearnSelectFromModel', convert_sklearn_feature_selection)
register_converter('SklearnSelectFwe', convert_sklearn_feature_selection)
register_converter('SklearnSelectKBest', convert_sklearn_feature_selection)
register_converter('SklearnSelectPercentile', convert_sklearn_feature_selection)
register_converter('SklearnVarianceThreshold', convert_sklearn_feature_selection)
