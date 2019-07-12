# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
from onnx.helper import make_tensor
from onnx import TensorProto
from sklearn.gaussian_process.kernels import (
    Sum, Product, ConstantKernel,
    RBF, DotProduct, ExpSineSquared,
    RationalQuadratic,
)
from ..algebra.complex_functions import onnx_squareform_pdist, onnx_cdist
from ..algebra.onnx_ops import (
    OnnxMul, OnnxMatMul, OnnxAdd,
    OnnxTranspose, OnnxDiv, OnnxExp,
    OnnxShape, OnnxSin, OnnxPow,
    OnnxReduceSum, OnnxSqueeze,
    OnnxIdentity, OnnxReduceSumSquare
)
try:
    from ..algebra.onnx_ops import OnnxConstantOfShape
except ImportError:
    OnnxConstantOfShape = None


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
        onnx_zeros = _zero_vector_of_size(X, keepdims=0)
        return OnnxAdd(onnx_zeros,
                       np.array([kernel.constant_value],
                                dtype=np.float32),
                       output_names=output_names)

    if isinstance(kernel, (RBF, ExpSineSquared, RationalQuadratic)):
        onnx_zeros = _zero_vector_of_size(X, keepdims=0)
        if isinstance(kernel, RBF):
            return OnnxAdd(onnx_zeros,
                           np.array([1],
                                    dtype=np.float32),
                           output_names=output_names)
        else:
            return OnnxAdd(onnx_zeros, np.array([1], dtype=np.float32),
                           output_names=output_names)

    if isinstance(kernel, DotProduct):
        t_sigma_0 = py_make_float_array(kernel.sigma_0 ** 2)
        return OnnxSqueeze(
            OnnxAdd(OnnxReduceSumSquare(X, axes=[1]), t_sigma_0),
            output_names=output_names, axes=[1])

    raise RuntimeError("Unable to convert diag method for "
                       "class {}.".format(type(kernel)))


def py_make_float_array(cst):
    return np.array([cst], dtype=np.float32)


def _convert_exp_sine_squared(X, Y, length_scale=1.2, periodicity=1.1,
                              pi=3.141592653589793, **kwargs):
    dists = onnx_cdist(X, Y, metric="euclidean")
    t_pi = py_make_float_array(pi)
    t_periodicity = py_make_float_array(periodicity)
    arg = OnnxMul(OnnxDiv(dists, t_periodicity), t_pi)
    sin_of_arg = OnnxSin(arg)
    t_2 = py_make_float_array(2)
    t__2 = py_make_float_array(-2)
    t_length_scale = py_make_float_array(length_scale)
    K = OnnxExp(OnnxMul(OnnxPow(OnnxDiv(sin_of_arg, t_length_scale),
                                t_2), t__2))
    return OnnxIdentity(K, **kwargs)


def _convert_dot_product(X, Y, sigma_0=2.0, **kwargs):
    """
    Implements the kernel
    :math:`k(x_i,x_j)=\\sigma_0^2+x_i\\cdot x_j`.
    """
    # It only works in two dimensions.
    t_sigma_0 = py_make_float_array(sigma_0 ** 2)
    K = OnnxAdd(OnnxMatMul(X, OnnxTranspose(Y, perm=[1, 0])),
                t_sigma_0)
    return OnnxIdentity(K, **kwargs)


def _convert_rational_quadratic(X, Y, length_scale=1.0, alpha=2.0, **kwargs):
    """
    Implements the kernel
    :math:`k(x_i,x_j)=(1 + d(x_i, x_j)^2 / (2*\\alpha * l^2))^{-\\alpha}`.
    """
    dists = onnx_cdist(X, Y, metric="sqeuclidean")
    cst = length_scale ** 2 * alpha * 2
    t_cst = py_make_float_array(cst)
    tmp = OnnxDiv(dists, t_cst)
    t_one = py_make_float_array(1)
    base = OnnxAdd(tmp, t_one)
    t_alpha = py_make_float_array(-alpha)
    K = OnnxPow(base, t_alpha)
    return OnnxIdentity(K, **kwargs)


def convert_kernel(kernel, X, output_names=None,
                   x_train=None):
    if isinstance(kernel, Sum):
        return OnnxAdd(convert_kernel(kernel.k1, X, x_train=x_train),
                       convert_kernel(kernel.k2, X, x_train=x_train),
                       output_names=output_names)
    if isinstance(kernel, Product):
        return OnnxMul(convert_kernel(kernel.k1, X, x_train=x_train),
                       convert_kernel(kernel.k2, X, x_train=x_train),
                       output_names=output_names)

    if isinstance(kernel, ConstantKernel):
        # X and x_train should have the same number of features.
        if x_train is None:
            onnx_zeros_x = _zero_vector_of_size(X, keepdims=1)
            onnx_zeros_y = onnx_zeros_x
        else:
            onnx_zeros_x = _zero_vector_of_size(X, keepdims=1)
            onnx_zeros_y = _zero_vector_of_size(x_train, keepdims=1)

        tr = OnnxTranspose(onnx_zeros_y, perm=[1, 0])
        mat = OnnxMatMul(onnx_zeros_x, tr)
        return OnnxAdd(mat,
                       np.array([kernel.constant_value],
                                dtype=np.float32),
                       output_names=output_names)

    if isinstance(kernel, RBF):
        if not isinstance(kernel.length_scale, (float, int)):
            raise NotImplementedError(
                "length_scale should be float not {}.".format(
                    type(kernel.length_scale)))

        # length_scale = np.squeeze(length_scale).astype(float)
        zeroh = _zero_vector_of_size(X, axis=1, keepdims=0)
        zerov = _zero_vector_of_size(X, axis=0, keepdims=1)

        tensor_value = make_tensor(
            "value", TensorProto.FLOAT, (1,), [kernel.length_scale])
        const = OnnxConstantOfShape(OnnxShape(zeroh),
                                    value=tensor_value)
        X_scaled = OnnxDiv(X, const)
        if x_train is None:
            dist = onnx_squareform_pdist(X_scaled, metric='sqeuclidean')
        else:
            x_train_scaled = OnnxDiv(x_train, const)
            dist = onnx_cdist(X_scaled, x_train_scaled, metric='sqeuclidean')

        tensor_value = make_tensor(
            "value", TensorProto.FLOAT, (1,), [-0.5])
        cst5 = OnnxConstantOfShape(OnnxShape(zerov), value=tensor_value)

        # K = np.exp(-.5 * dists)
        exp = OnnxExp(OnnxMul(dist, cst5), output_names=output_names)

        # This should not be needed.
        # K = squareform(K)
        # np.fill_diagonal(K, 1)
        return exp

    if isinstance(kernel, ExpSineSquared):
        if not isinstance(kernel.length_scale, (float, int)):
            raise NotImplementedError(
                "length_scale should be float not {}.".format(
                    type(kernel.length_scale)))

        return _convert_exp_sine_squared(
            X, Y=X if x_train is None else x_train,
            length_scale=kernel.length_scale,
            periodicity=kernel.periodicity,
            output_names=output_names)

    if isinstance(kernel, DotProduct):
        if not isinstance(kernel.sigma_0, (float, int)):
            raise NotImplementedError(
                "sigma_0 should be float not {}.".format(
                    type(kernel.sigma_0)))

        if x_train is None:
            return _convert_dot_product(X, X, sigma_0=kernel.sigma_0,
                                        output_names=output_names)
        else:
            if len(x_train.shape) != 2:
                raise NotImplementedError(
                    "Only DotProduct for two dimension train set is "
                    "implemented.")
            return _convert_dot_product(X, x_train, sigma_0=kernel.sigma_0,
                                        output_names=output_names)

    if isinstance(kernel, RationalQuadratic):
        if x_train is None:
            return _convert_rational_quadratic(
                X, X, length_scale=kernel.length_scale,
                alpha=kernel.alpha, output_names=output_names)
        else:
            return _convert_rational_quadratic(
                X, x_train, length_scale=kernel.length_scale,
                alpha=kernel.alpha, output_names=output_names)

    raise RuntimeError("Unable to convert __call__ method for "
                       "class {}.".format(type(kernel)))


def _zero_vector_of_size(X, output_names=None, axis=0, keepdims=None):
    if keepdims is None:
        raise ValueError("Default for keepdims is not allowed.")
    res = OnnxReduceSum(OnnxConstantOfShape(OnnxShape(X)),
                        axes=[1-axis], keepdims=keepdims,
                        output_names=output_names)
    return res
