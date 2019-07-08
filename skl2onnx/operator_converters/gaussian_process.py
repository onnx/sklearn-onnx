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
    RBF, ConstantKernel as C, DotProduct,
    ExpSineSquared
)
from ..common._registration import register_converter
from ..algebra.complex_functions import squareform_pdist, cdist, inner
from ..algebra.onnx_ops import (
    OnnxMul, OnnxMatMul, OnnxAdd, OnnxSqrt,
    OnnxTranspose, OnnxDiv, OnnxExp,
    OnnxShape, OnnxSin, OnnxPow,
    OnnxReduceSum, OnnxSqueeze
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
            dist = squareform_pdist(X_scaled, metric='sqeuclidean')
        else:
            x_train_scaled = OnnxDiv(x_train, const)
            dist = cdist(X_scaled, x_train_scaled, metric='sqeuclidean')

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
            dist = squareform_pdist(X_scaled, metric='sqeuclidean')
        else:
            x_train_scaled = OnnxDiv(x_train, const)
            dist = cdist(X_scaled, x_train_scaled, metric='sqeuclidean')

        tensor_value = make_tensor("value", TensorProto.FLOAT, (1,),
                                   [np.pi / kernel.periodicity])
        piper = OnnxSqueeze(OnnxConstantOfShape(OnnxShape(zerov),
                                                value=tensor_value),
                            axes=[1])

        # arg = np.pi * dists / self.periodicity
        # sin_of_arg = np.sin(arg)
        sinm = OnnxSin(OnnxMul(dist, piper))

        tensor_value = make_tensor("value", TensorProto.FLOAT, (1,),
                                   [1. / kernel.length_scale])
        scale = OnnxSqueeze(OnnxConstantOfShape(OnnxShape(zerov),
                                                value=tensor_value),
                            axes=[1])

        if 'cst2' in context:
            cst2 = context['cst2']
        else:
            tensor_value = make_tensor("value", TensorProto.FLOAT, (1,), [2.])
            cst2 = OnnxSqueeze(OnnxConstantOfShape(OnnxShape(zerov),
                                                   value=tensor_value),
                               axes=[1])
        if 'cst_2' in context:
            cst_2 = context['cst_2']
        else:
            tensor_value = make_tensor("value", TensorProto.FLOAT, (1,), [-2.])
            cst_2 = OnnxSqueeze(OnnxConstantOfShape(OnnxShape(zerov),
                                                    value=tensor_value),
                                axes=[1])
        scaled = OnnxMul(OnnxPow(OnnxMul(sinm, scale), cst2), cst_2)

        exp = OnnxExp(scaled, output_names=output_names)

        # This should not be needed.
        # K = squareform(K)
        # np.fill_diagonal(K, 1)
        return exp

    if isinstance(kernel, DotProduct):
        if isinstance(kernel.sigma_0, (int, float)):
            if x_train is None:
                dot = inner(X, X)
                tensor_value = make_tensor(
                    "value", TensorProto.FLOAT, (1,), [kernel.sigma_0 ** 2])
                cst = OnnxConstantOfShape(OnnxShape(X), value=tensor_value)
                add = OnnxAdd(dot, cst)
            else:
                dot = inner(X, x_train)
                add = OnnxAdd(dot, np.full((1, x_train.shape[1]),
                                           kernel.sigma_0 ** 2,
                                           dtype=np.float32))
        else:
            raise NotImplementedError("Not implemented yet for type {}"
                                      "".format(type(kernel.sigma_0)))
        return add

    raise RuntimeError("Unable to convert __call__ method for "
                       "class {}.".format(type(kernel)))


def _zero_vector_of_size(X, output_names=None, axis=0):
    res = OnnxReduceSum(OnnxConstantOfShape(OnnxShape(X)),
                        axes=[1-axis],
                        output_names=output_names)
    return res


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
