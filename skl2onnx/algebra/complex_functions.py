# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from collections import OrderedDict
import numpy as np
from ..common.data_types import FloatTensorType, DoubleTensorType
from .onnx_ops import (
    OnnxIdentity, OnnxScan, OnnxTranspose,
    OnnxSub, OnnxReduceSumSquare, OnnxSqueezeApi11,
    OnnxSqrt, OnnxPow, OnnxAbs, OnnxReduceSumApi11
)


def onnx_squareform_pdist(X, metric='sqeuclidean', dtype=None,
                          op_version=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric=metric))``.
    """
    if metric == 'sqeuclidean':
        return _onnx_squareform_pdist_sqeuclidean(
            X, dtype=dtype, op_version=op_version, **kwargs)
    elif metric == 'euclidean':
        res = _onnx_squareform_pdist_sqeuclidean(
            X, dtype=dtype, op_version=op_version)
        return OnnxSqrt(res, op_version=op_version, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def _onnx_squareform_pdist_sqeuclidean(X, dtype=None, op_version=None,
                                       **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric='sqeuclidean'))``.
    """
    diff = OnnxSub('next_in', 'next', output_names=['diff'],
                   op_version=op_version)
    id_next = OnnxIdentity('next_in', output_names=['next_out'],
                           op_version=op_version)
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1],
                               op_version=op_version)
    flat = OnnxSqueezeApi11(norm, output_names=['scan_out'], axes=[1],
                            op_version=op_version)
    tensor_type = FloatTensorType if dtype == np.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('pdistsqe')
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', tensor_type()),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        target_opset=op_version)

    node = OnnxScan(X, X, output_names=['u(scan0)', 'u(scan1)'],
                    num_scan_inputs=1, body=scan_body.graph,
                    op_version=op_version, **kwargs)
    return node[1]


def onnx_cdist(XA, XB, metric='sqeuclidean', dtype=None,
               op_version=None, dim_in=None, dim_out=None,
               **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(XA, XB, metric=metric)``.

    :param XA: array or OnnxOperatorMixin
    :param XB: array or OnnxOperatorMixin
    :param metric: distance type
    :param dtype: *np.float32* or *np.float64*
    :param op_version: opset version
    :param dim_in: dimension of the input vectorial space
        (if known)
    :param dim_out: dimension of the output vectorial space
        (if known)
    :param kwargs: addition parameter
    :return: OnnxOperatorMixin
    """
    if metric == 'sqeuclidean':
        return _onnx_cdist_sqeuclidean(
            XA, XB, dtype=dtype, op_version=op_version,
            dim_in=dim_in, dim_out=dim_out, **kwargs)
    elif metric == 'euclidean':
        res = _onnx_cdist_sqeuclidean(
            XA, XB, dtype=dtype, op_version=op_version,
            dim_in=dim_in, dim_out=dim_out)
        return OnnxSqrt(res, op_version=op_version, **kwargs)
    elif metric == 'minkowski':
        p = kwargs.pop('p')
        res = _onnx_cdist_minkowski(
            XA, XB, dtype=dtype, op_version=op_version, p=p,
            dim_in=dim_in, dim_out=dim_out)
        return OnnxPow(res, np.array([1. / p], dtype=dtype),
                       op_version=op_version, **kwargs)
    elif metric in ('manhattan', 'cityblock'):
        return _onnx_cdist_manhattan(
            XA, XB, dtype=dtype, op_version=op_version,
            dim_in=dim_in, dim_out=dim_out, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def _onnx_cdist_begin(op_version):
    diff = OnnxSub('next_in', 'next', output_names=[
                   'diff'], op_version=op_version)
    id_next = OnnxIdentity('next_in', output_names=[
                           'next_out'], op_version=op_version)
    return diff, id_next


def _onnx_cdist_end(XA, XB, id_next, flat, dtype, op_version,
                    dim_in=None, dim_out=None, **kwargs):
    tensor_type = FloatTensorType if dtype == np.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('cdistd')
    shape_in = (tensor_type() if dim_in is None
                else tensor_type([None, dim_in]))
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', shape_in),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        target_opset=op_version)

    node = OnnxScan(XA, XB, output_names=['u(scan0)', 'u(scan1)'],
                    num_scan_inputs=1, body=scan_body.graph,
                    op_version=op_version)
    return OnnxTranspose(node[1], perm=[1, 0], op_version=op_version,
                         **kwargs)


def _onnx_cdist_sqeuclidean(XA, XB, dtype=None, op_version=None,
                            dim_in=None, dim_out=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, metric='sqeuclidean')``.
    """
    diff, id_next = _onnx_cdist_begin(op_version)
    norm = OnnxReduceSumSquare(
        diff, output_names=['norm'], axes=[1],
        keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=['scan_out'], op_version=op_version)
    return _onnx_cdist_end(XA, XB, id_next, flat, dtype, op_version,
                           dim_in=dim_in, dim_out=dim_out, **kwargs)


def _onnx_cdist_minkowski(XA, XB, dtype=None, op_version=None, p=2,
                          dim_in=None, dim_out=None, **kwargs):
    """
    Returns the ONNX graph which computes the Minkowski distance
    or ``minkowski(XA, XB, p)``.
    """
    diff, id_next = _onnx_cdist_begin(op_version)
    diff_pow = OnnxPow(OnnxAbs(diff, op_version=op_version),
                       np.array([p], dtype=dtype), op_version=op_version)
    norm = OnnxReduceSumApi11(
        diff_pow, axes=[1], output_names=['norm'],
        keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=['scan_out'], op_version=op_version)
    return _onnx_cdist_end(XA, XB, id_next, flat, dtype, op_version,
                           dim_in=dim_in, dim_out=dim_out, **kwargs)


def _onnx_cdist_manhattan(XA, XB, dtype=None, op_version=None,
                          dim_in=None, dim_out=None, **kwargs):
    """
    Returns the ONNX graph which computes the Manhattan distance
    or ``Manhattan(X, Y)``.
    """
    diff, id_next = _onnx_cdist_begin(op_version)
    diff_pow = OnnxAbs(diff, op_version=op_version)
    norm = OnnxReduceSumApi11(diff_pow, axes=[1], output_names=[
                         'norm'], keepdims=0, op_version=op_version)
    flat = OnnxIdentity(norm, output_names=['scan_out'], op_version=op_version)
    return _onnx_cdist_end(XA, XB, id_next, flat, dtype, op_version,
                           dim_in=dim_in, dim_out=dim_out, **kwargs)
