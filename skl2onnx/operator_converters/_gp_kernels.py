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
    RBF, DotProduct, ExpSineSquared
)
from ..algebra.complex_functions import onnx_squareform_pdist, onnx_cdist
from ..algebra.onnx_ops import (
    OnnxMul, OnnxMatMul, OnnxAdd,
    OnnxTranspose, OnnxDiv, OnnxExp,
    OnnxShape, OnnxSin, OnnxPow,
    OnnxReduceSum, OnnxSqueeze,
    OnnxIdentity
)
try:
    from ..algebra.onnx_ops import OnnxConstantOfShape
except ImportError:
    OnnxConstantOfShape = None


def convert_kernel_diag(context, kernel, X, output_names=None):
    if isinstance(kernel, Sum):
        return OnnxAdd(convert_kernel_diag(context, kernel.k1, X),
                       convert_kernel_diag(context, kernel.k2, X),
                       output_names=output_names)
    if isinstance(kernel, Product):
        return OnnxMul(convert_kernel_diag(context, kernel.k1, X),
                       convert_kernel_diag(context, kernel.k2, X),
                       output_names=output_names)
    if isinstance(kernel, ConstantKernel):
        if 'zerov' in context:
            onnx_zeros = context['zerov']
        else:
            onnx_zeros = _zero_vector_of_size(X)
            context['zerov'] = onnx_zeros
        return OnnxAdd(onnx_zeros,
                       np.array([kernel.constant_value],
                                dtype=np.float32),
                       output_names=output_names)
    if isinstance(kernel, RBF):
        if 'zerov' in context:
            onnx_zeros = context['zerov']
        else:
            onnx_zeros = _zero_vector_of_size(X)
            context['zerov'] = onnx_zeros
        return OnnxAdd(onnx_zeros,
                       np.array([1],
                                dtype=np.float32),
                       output_names=output_names)
    raise RuntimeError("Unable to convert diag method for "
                       "class {}.".format(type(kernel)))


def py_make_float_array(cst):
    return np.array([cst], dtype=np.float32)


def _convert_exp_sine_squared_none(X, length_scale=1.2, periodicity=1.1,
                                   pi=3.141592653589793, **kwargs):
    dists = onnx_squareform_pdist(X, metric="euclidean")
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


def convert_kernel(context, kernel, X, output_names=None,
                   x_train=None):
    if isinstance(kernel, Sum):
        return OnnxAdd(convert_kernel(context, kernel.k1, X, x_train=x_train),
                       convert_kernel(context, kernel.k2, X, x_train=x_train),
                       output_names=output_names)
    if isinstance(kernel, Product):
        return OnnxMul(convert_kernel(context, kernel.k1, X, x_train=x_train),
                       convert_kernel(context, kernel.k2, X, x_train=x_train),
                       output_names=output_names)

    if isinstance(kernel, ConstantKernel):
        # X and x_train should have the same number of features.
        if 'zerov' in context:
            onnx_zeros = context['zerov']
        else:
            onnx_zeros = _zero_vector_of_size(X)
            context['zerov'] = onnx_zeros

        tr = OnnxTranspose(onnx_zeros, perm=[1, 0])
        mat = OnnxMatMul(onnx_zeros, tr)
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
        if 'zeroh' in context:
            zeroh = context['zeroh']
        else:
            zeroh = _zero_vector_of_size(X, axis=1)
            context['zeroh'] = zeroh

        if 'zerov' in context:
            zerov = context['zerov']
        else:
            zerov = _zero_vector_of_size(X, axis=0)
            context['zerov'] = zeroh

        tensor_value = make_tensor(
            "value", TensorProto.FLOAT, (1,), [kernel.length_scale])
        const = OnnxSqueeze(OnnxConstantOfShape(OnnxShape(zeroh),
                                                value=tensor_value),
                            axes=[0])
        X_scaled = OnnxDiv(X, const)
        if x_train is None:
            dist = onnx_squareform_pdist(X_scaled, metric='sqeuclidean')
        else:
            x_train_scaled = OnnxDiv(x_train, const)
            dist = onnx_cdist(X_scaled, x_train_scaled, metric='sqeuclidean')

        if 'cst5' in context:
            cst5 = context['cst5']
        else:
            tensor_value = make_tensor("value", TensorProto.FLOAT, (1,), [-.5])
            cst5 = OnnxSqueeze(OnnxConstantOfShape(OnnxShape(zerov),
                                                   value=tensor_value),
                               axes=[1])
            context['cst5'] = cst5

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

        if x_train is None:
            return _convert_exp_sine_squared_none(
                X, length_scale=kernel.length_scale,
                periodicity=kernel.periodicity,
                output_names=output_names)
        else:
            return _convert_exp_sine_squared(
                X, x_train, length_scale=kernel.length_scale,
                periodicity=kernel.periodicity,
                output_names=output_names)

    if isinstance(kernel, DotProduct):
        if not isinstance(kernel.sigma_0, (float, int)):
            raise NotImplementedError(
                "sigma_0 should be float not {}.".format(
                    type(kernel.length_scale)))

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

    raise RuntimeError("Unable to convert __call__ method for "
                       "class {}.".format(type(kernel)))


def _zero_vector_of_size(X, output_names=None, axis=0):
    res = OnnxReduceSum(OnnxConstantOfShape(OnnxShape(X)),
                        axes=[1-axis],
                        output_names=output_names)
    return res
