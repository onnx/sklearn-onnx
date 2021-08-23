# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..algebra.onnx_ops import OnnxConcat, OnnxReshape
from ..algebra.onnx_operator import OnnxSubEstimator


def convert_multi_output_regressor_regressor(
        scope: Scope, operator: Operator, container: ModelComponentContainer):
    """
    Converts a *MultiOutputRegressor* into *ONNX* format.
    """
    op_version = container.target_opset
    op = operator.raw_operator
    inp = operator.inputs[0]
    y_list = [
        OnnxReshape(
            OnnxSubEstimator(sub, inp, op_version=op_version),
            np.array([-1, 1], dtype=np.int64),
            op_version=op_version)
        for sub in op.estimators_]

    output = OnnxConcat(*y_list, axis=1, op_version=op_version,
                        output_names=[operator.outputs[0]])
    output.add_to(scope=scope, container=container)


register_converter('SklearnMultiOutputRegressor',
                   convert_multi_output_regressor_regressor)
