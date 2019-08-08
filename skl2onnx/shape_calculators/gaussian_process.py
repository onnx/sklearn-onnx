# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, DoubleTensorType
from ..common.utils import check_input_and_output_types


def calculate_sklearn_gaussian_process_regressor_shape(operator):
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, DoubleTensorType],
        good_output_types=[FloatTensorType, DoubleTensorType])
    if len(operator.inputs) != 1:
        raise RuntimeError("Only one input vector is allowed for "
                           "GaussianProcessRegressor.")
    if len(operator.outputs) not in (1, 2):
        raise RuntimeError("One output is expected for "
                           "GaussianProcessRegressor.")

    variable = operator.inputs[0]

    N = variable.type.shape[0] if len(variable.type.shape) > 0 else 1
    op = operator.raw_operator

    # Output 1 is mean
    # Output 2 is cov or std
    if hasattr(op, 'y_train_') and op.y_train_ is not None:
        dim = 1 if len(op.y_train_.shape) == 1 else op.y_train_.shape[1]
    else:
        dim = 1
    operator.outputs[0].type.shape = [N, dim]


register_shape_calculator('SklearnGaussianProcessRegressor',
                          calculate_sklearn_gaussian_process_regressor_shape)
