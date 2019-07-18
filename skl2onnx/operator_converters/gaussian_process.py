# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from ..proto import onnx_proto
from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSqrt, OnnxMatMul, OnnxSub, OnnxReduceSum,
    OnnxMul, OnnxMax, OnnxCast
)
from ..algebra.onnx_operator import OnnxOperator
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
    """
    The method *predict* from class *GaussianProcessRegressor*
    may cache some results if it is called with parameter
    ``return_std=True`` or ``return_cov=True``. This converter
    needs to be called with theses options to enable
    the second results.

    Last option is ``float64=True``. The converted
    model may have too many discrepencies if float32
    are used. The conversion may happen fully with float64
    or partially if this option is set up. In that case,
    a few operator only will be used running with float64.
    """
    dtype = container.dtype
    if dtype is None:
        raise RuntimeError("dtype cannot be None")
    X = operator.inputs[0]
    out = operator.outputs
    op = operator.raw_operator

    options = container.get_options(op, dict(return_cov=False,
                                             return_std=False,
                                             float64=False))
    if hasattr(op, 'kernel_') and op.kernel_ is not None:
        kernel = op.kernel_
    elif op.kernel is None:
        kernel = (C(1.0, constant_value_bounds="fixed") *
                  RBF(1.0, length_scale_bounds="fixed"))
    else:
        kernel = op.kernel

    if not hasattr(op, "X_train_") or op.X_train_ is None:
        out0 = _zero_vector_of_size(X, keepdims=1, output_names=out[:1],
                                    dtype=dtype)

        outputs = [out0]
        if options['return_cov']:
            outputs.append(convert_kernel(kernel, X,
                                          output_names=out[1:],
                                          dtype=dtype,
                                          try_float64=options['float64']))
        if options['return_std']:
            outputs.append(OnnxSqrt(convert_kernel_diag(
                                        kernel, X, dtype=dtype,
                                        try_float64=options['float64']),
                                    output_names=out[1:]))
    else:
        out0 = _zero_vector_of_size(X, keepdims=1, dtype=dtype)

        # Code scikit-learn
        # K_trans = self.kernel_(X, self.X_train_)
        # y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        # y_mean = self._y_train_mean + y_mean  # undo normal.

        k_trans = convert_kernel(kernel, X,
                                 x_train=op.X_train_.astype(dtype),
                                 dtype=dtype,
                                 try_float64=options['float64'])
        k_trans.set_onnx_name_prefix('kgpd')

        if options['float64']:
            if dtype == np.float64:
                raise RuntimeError(
                    "Redundant option. Option float64 should not be "
                    "defined if dtype=float64.")
            k_trans64 = OnnxCast(k_trans, to=onnx_proto.TensorProto.DOUBLE)
            y_mean_b_64 = OnnxMatMul(k_trans64,
                                     OnnxOperator.ConstantVariable(
                                        op.alpha_.astype(np.float64),
                                        implicit_cast=False))
            y_mean_b = OnnxCast(y_mean_b_64, to=onnx_proto.TensorProto.FLOAT)
        else:
            y_mean_b = OnnxMatMul(k_trans, op.alpha_.astype(dtype))

        mean_y = op._y_train_mean.astype(dtype)
        if len(mean_y.shape) == 1:
            mean_y = mean_y.reshape(mean_y.shape + (1,))
        y_mean = OnnxAdd(y_mean_b, mean_y,
                         output_names=out[:1])
        y_mean.set_onnx_name_prefix('gpr')
        outputs = [y_mean]

        if options['return_cov']:
            raise NotImplementedError()
        if options['return_std']:
            if op._K_inv is None:
                raise RuntimeError(
                    "The method *predict* must be called once with parameter "
                    "return_std=True to compute internal variables. "
                    "They cannot be computed here as the same operation "
                    "(matrix inversion) produces too many discrepencies "
                    "if done with single floats than double floats.")
            _K_inv = op._K_inv

            # y_var = self.kernel_.diag(X)
            y_var = convert_kernel_diag(kernel, X, dtype=dtype,
                                        try_float64=options['float64'])

            # y_var -= np.einsum("ij,ij->i",
            #       np.dot(K_trans, self._K_inv), K_trans)
            k_dot = OnnxMatMul(k_trans, _K_inv.astype(dtype))
            ys_var = OnnxSub(y_var,
                             OnnxReduceSum(OnnxMul(k_dot, k_trans),
                                           axes=[1], keepdims=0))

            # y_var_negative = y_var < 0
            # if np.any(y_var_negative):
            #     y_var[y_var_negative] = 0.0
            ys0_var = OnnxMax(ys_var, np.array([0], dtype=dtype))

            # var = np.sqrt(ys0_var)
            var = OnnxSqrt(ys0_var, output_names=out[1:])
            var.set_onnx_name_prefix('gprv')
            outputs.append(var)

    for o in outputs:
        o.add_to(scope, container)


if OnnxConstantOfShape is not None:
    register_converter('SklearnGaussianProcessRegressor',
                       convert_gaussian_process_regressor)
