# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSqrt, OnnxMatMul
)
try:
    from ..algebra.onnx_ops import OnnxConstantOfShape
except ImportError:
    OnnxConstantOfShape = None

from ._gp_kernels import (
    convert_kernel_diag,
    convert_kernel,
    _zero_vector_of_size
)


def convert_gaussian_process_regressor(scope, operator, container):

    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator

    options = container.get_options(op, dict(return_cov=False,
                                             return_std=False))
    if op.kernel is None:
        kernel = (C(1.0, constant_value_bounds="fixed") *
                  RBF(1.0, length_scale_bounds="fixed"))
    else:
        kernel = op.kernel

    if not hasattr(op, "X_train_"):
        out0 = _zero_vector_of_size(X, output_names=out[:1])
        context = dict(zerov=out0)

        outputs = [out0]
        if options['return_cov']:
            outputs.append(convert_kernel(context, kernel, X,
                                          output_names=out[1:]))
        if options['return_std']:
            outputs.append(OnnxSqrt(convert_kernel_diag(context, kernel, X),
                                    output_names=out[1:]))
    else:
        out0 = _zero_vector_of_size(X)
        context = dict(zerov=out0)

        # Code scikit-learn
        # K_trans = self.kernel_(X, self.X_train_)
        # y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        # y_mean = self._y_train_mean + y_mean  # undo normal.

        k_trans = convert_kernel(context, kernel, X,
                                 x_train=op.X_train_.astype(np.float32))
        y_mean_b = OnnxMatMul(k_trans, op.alpha_.astype(np.float32))
        y_mean = OnnxAdd(y_mean_b, op._y_train_mean.astype(np.float32),
                         output_names=out[:1])
        outputs = [y_mean]

        if options['return_cov']:
            raise NotImplementedError()
        if options['return_std']:
            raise NotImplementedError()

    for o in outputs:
        o.add_to(scope, container)


if OnnxConstantOfShape is not None:
    register_converter('SklearnGaussianProcessRegressor',
                       convert_gaussian_process_regressor)
