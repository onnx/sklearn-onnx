# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .onnx_operator import OnnxOperator


class OnnxCDist(OnnxOperator):
    """
    Defines a custom operator not defined by ONNX
    specifications but in onnxruntime.
    """

    since_version = 1
    expected_inputs = ['X', 'Y']
    expected_outputs = ['dist']
    input_range = [2, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'com.microsoft'
    operator_name = 'CDist'
    past_version = {}

    def __init__(self, X, Y, metric='sqeuclidean', op_version=None,
                 **kwargs):
        """
        :param X: array or OnnxOperatorMixin
        :param Y: array or OnnxOperatorMixin
        :param metric: distance type
        :param dtype: *np.float32* or *np.float64*
        :param op_version: opset version
        :param kwargs: addition parameter
        """
        OnnxOperator.__init__(self, X, Y, metric=metric,
                              op_version=op_version, **kwargs)


class OnnxSolve(OnnxOperator):
    """
    Defines a custom operator not defined by ONNX
    specifications but in onnxruntime.
    """

    since_version = 1
    expected_inputs = ['A', 'Y']
    expected_outputs = ['X']
    input_range = [2, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'com.microsoft'
    operator_name = 'Solve'
    past_version = {}

    def __init__(self, A, Y, lower=False, transposed=False,
                 op_version=None, **kwargs):
        """
        :param A: array or OnnxOperatorMixin
        :param Y: array or OnnxOperatorMixin
        :param lower: see :epkg:`solve`
        :param transposed: see :epkg:`solve`
        :param dtype: *np.float32* or *np.float64*
        :param op_version: opset version
        :param kwargs: additional parameters
        """
        OnnxOperator.__init__(self, A, Y,
                              lower=lower, transposed=transposed,
                              op_version=op_version, **kwargs)
