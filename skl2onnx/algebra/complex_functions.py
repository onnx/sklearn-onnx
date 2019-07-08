# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from collections import OrderedDict
from ..common.data_types import FloatTensorType
from .onnx_ops import (
    OnnxIdentity, OnnxScan, OnnxTranspose,
    OnnxSub, OnnxReduceSumSquare, OnnxSqueeze,
    OnnxMatMul, OnnxReduceSum, OnnxSqrt
)


def squareform_pdist(X, metric='sqeuclidean', **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric=metric))``.
    """
    if metric == 'sqeuclidean':
        return _squareform_pdist_sqeuclidean(X, **kwargs)
    elif metric == 'euclidean':
        res = _squareform_pdist_sqeuclidean(X)
        return OnnxSqrt(res, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def cdist(X, Y, metric='sqeuclidean', **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, Y, metric=metric)``.
    """
    if metric == 'sqeuclidean':
        return _cdist_sqeuclidean(X, Y, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def _squareform_pdist_sqeuclidean(X, **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric='sqeuclidean'))``.
    """
    diff = OnnxSub('next_in', 'next', output_names=['diff'])
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1])
    flat = OnnxSqueeze(norm, output_names=['scan_out'], axes=[1])
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', FloatTensorType()),
                     ('next', FloatTensorType())]),
        outputs=[('next_out', FloatTensorType()),
                 ('scan_out', FloatTensorType())],
        other_outputs=[flat])

    node = OnnxScan(X, X, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph,
                    **kwargs)
    return node[1]


def _cdist_sqeuclidean(X, Y, **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, metric='sqeuclidean')``.
    """
    diff = OnnxSub('next_in', 'next', output_names=['diff'])
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1])
    flat = OnnxSqueeze(norm, output_names=['scan_out'], axes=[1])
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', FloatTensorType()),
                     ('next', FloatTensorType())]),
        outputs=[('next_out', FloatTensorType()),
                 ('scan_out', FloatTensorType())],
        other_outputs=[flat])

    node = OnnxScan(X, Y, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph)
    return OnnxTranspose(node[1], perm=[1, 0],
                         **kwargs)


def inner(X, Y, **kwargs):
    """
    Returns the ONNX graph which computes
    ``inner(X, X)`` only for 2D matrices.
    """
    mm = OnnxMatMul('next_in', 'next', output_names=['inner1'])
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    inn = OnnxReduceSum(mm, output_names=['inner'], axes=[1])
    flat = OnnxSqueeze(inn, output_names=['scan_out'], axes=[1])
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', FloatTensorType()),
                     ('next', FloatTensorType())]),
        outputs=[('next_out', FloatTensorType()),
                 ('scan_out', FloatTensorType())],
        other_outputs=[flat])

    node = OnnxScan(X, Y, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph,
                    scan_output_axes=[1],
                    **kwargs)
    return node[1]
