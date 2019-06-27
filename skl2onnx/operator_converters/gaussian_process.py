# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxMul, OnnxMatMul, OnnxAdd, OnnxSqrt
)
from sklearn.gaussian_process.kernels import Sum, Product, ConstantKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def convert_kernel_diag(kernel, X, output_names=None):
    if isinstance(kernel, Sum):
        return OnnxAdd(convert_kernel_diag(kernel.k1, X),
                       convert_kernel_diag(kernel.k2, X),
                       output_names=output_names)
    if isinstance(kernel, Product):
        return OnnxMul(convert_kernel_diag(kernel.k1, X),
                       convert_kernel_diag(kernel.k2, X),
                       output_names=output_names)
    if isinstance(kernel, ConstantKernel):
        zeros = np.zeros((X.type.shape[1], 1))
        onnx_zeros = OnnxMatMul(X, zeros)
        return OnnxAdd(onnx_zeros,
                       np.array([kernel.constant_value],
                                dtype=np.float32),
                       output_names=output_names)
    if isinstance(kernel, RBF):
        zeros = np.zeros((X.type.shape[1], 1))
        onnx_zeros = OnnxMatMul(X, zeros)
        return OnnxAdd(onnx_zeros,
                       np.array([1],
                                dtype=np.float32),
                       output_names=output_names)
    raise RuntimeError("Unable to convert diag function for "
                       "class {}.".format(type(kernel)))


def convert_kernel(kernel, X, output_names=None):
    raise NotImplementedError()


def convert_gaussian_process_regressor(scope, operator, container):

    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator

    options = container.get_options(op, dict(return_cov=False,
                                             return_std=False))

    if not hasattr(op, "X_train_"):
        if op.kernel is None:
            kernel = (C(1.0, constant_value_bounds="fixed") *
                      RBF(1.0, length_scale_bounds="fixed"))
        else:
            kernel = op.kernel

        y_mean = np.zeros((X.type.shape[1], 1))

        if options['return_cov']:
            out = [OnnxMatMul(X, y_mean, output_names=out[:1]),
                   convert_kernel(kernel, X, output_names=out[1:])]
        elif options['return_std']:
            out = [OnnxMatMul(X, y_mean, output_names=out[:1]),
                   OnnxSqrt(convert_kernel_diag(kernel, X),
                            output_names=out[1:])]
        else:
            out = [OnnxMatMul(X, y_mean, output_names=out[:1])]
    else:
        raise RuntimeError("The converter does not handle fitted "
                           "gaussian processes yet.")

    for o in out:
        o.add_to(scope, container)


register_converter('SklearnGaussianProcessRegressor',
                   convert_gaussian_process_regressor)
