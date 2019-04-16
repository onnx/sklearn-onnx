# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ._apply_operation import apply_cast, apply_reshape
from ..proto import onnx_proto


def _finalize_converter_classes(scope, argmax_output_name, output_full_name,
                                container, classes):
    """
    See :func:`convert_voting_classifier`.
    """
    if np.issubdtype(classes.dtype, np.floating):
        class_type = onnx_proto.TensorProto.INT32
        classes = np.array(list(map(lambda x: int(x), classes)))
    elif np.issubdtype(classes.dtype, np.signedinteger):
        class_type = onnx_proto.TensorProto.INT32
    else:
        classes = np.array([s.encode('utf-8') for s in classes])
        class_type = onnx_proto.TensorProto.STRING

    classes_name = scope.get_unique_variable_name('classes')
    container.add_initializer(classes_name, class_type, classes.shape, classes)

    array_feature_extractor_result_name = scope.get_unique_variable_name(
                                            'array_feature_extractor_result')
    container.add_node(
        'ArrayFeatureExtractor', [classes_name, argmax_output_name],
        array_feature_extractor_result_name, op_domain='ai.onnx.ml',
        name=scope.get_unique_operator_name('ArrayFeatureExtractor'))

    output_shape = (-1,)
    if class_type == onnx_proto.TensorProto.INT32:
        cast2_result_name = scope.get_unique_variable_name('cast2_result')
        reshaped_result_name = scope.get_unique_variable_name(
                                                'reshaped_result')
        apply_cast(scope, array_feature_extractor_result_name,
                   cast2_result_name, container,
                   to=onnx_proto.TensorProto.FLOAT)
        apply_reshape(scope, cast2_result_name, reshaped_result_name,
                      container, desired_shape=output_shape)
        apply_cast(scope, reshaped_result_name, output_full_name, container,
                   to=onnx_proto.TensorProto.INT64)
    else:  # string labels
        apply_reshape(scope, array_feature_extractor_result_name,
                      output_full_name, container, desired_shape=output_shape)
