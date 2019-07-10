# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSqrt, OnnxMatMul, OnnxSub, OnnxReduceSum,
    OnnxMul
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
            if op._K_inv is None:
                L_inv = solve_triangular(op.L_.T,
                                         np.eye(op.L_.shape[0]))
                _K_inv = L_inv.dot(L_inv.T)
            else:
                _K_inv = op._K_inv

            # y_var = self.kernel_.diag(X)
            y_var = convert_kernel_diag(context, op.kernel_, X)

            # y_var -= np.einsum("ij,ij->i",
            #       np.dot(K_trans, self._K_inv), K_trans)
            k_dot = OnnxMatMul(k_trans, _K_inv.astype(np.float32))
            ys_var = OnnxSub(y_var,
                             OnnxReduceSum(OnnxMul(k_dot, k_trans), axes=[1]))

            # skips next part
            # y_var_negative = y_var < 0
            # if np.any(y_var_negative):
            #     y_var[y_var_negative] = 0.0

            # var = np.sqrt(y_var)
            var = OnnxSqrt(ys_var, output_names=out[1:])
            outputs.append(var)

    for o in outputs:
        o.add_to(scope, container)


if OnnxConstantOfShape is not None:
    register_converter('SklearnGaussianProcessRegressor',
                       convert_gaussian_process_regressor)
