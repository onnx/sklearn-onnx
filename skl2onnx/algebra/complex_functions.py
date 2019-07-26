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
    OnnxSub, OnnxReduceSumSquare, OnnxSqueeze,
    OnnxSqrt
)


def onnx_squareform_pdist(X, metric='sqeuclidean', dtype=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric=metric))``.
    """
    if metric == 'sqeuclidean':
        return _onnx_squareform_pdist_sqeuclidean(X, dtype=dtype, **kwargs)
    elif metric == 'euclidean':
        res = _onnx_squareform_pdist_sqeuclidean(X, dtype=dtype)
        return OnnxSqrt(res, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def onnx_cdist(X, Y, metric='sqeuclidean', dtype=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, Y, metric=metric)``.
    """
    if metric == 'sqeuclidean':
        return _onnx_cdist_sqeuclidean(X, Y, dtype=dtype, **kwargs)
    elif metric == 'euclidean':
        res = _onnx_cdist_sqeuclidean(X, Y, dtype=dtype)
        return OnnxSqrt(res, **kwargs)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def _onnx_squareform_pdist_sqeuclidean(X, dtype=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(pdist(X, metric='sqeuclidean'))``.
    """
    diff = OnnxSub('next_in', 'next', output_names=['diff'])
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1])
    flat = OnnxSqueeze(norm, output_names=['scan_out'], axes=[1])
    tensor_type = FloatTensorType if dtype == np.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('pdistsqe')
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', tensor_type()),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        dtype=dtype)

    node = OnnxScan(X, X, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph,
                    **kwargs)
    return node[1]


def _onnx_cdist_sqeuclidean(X, Y, dtype=None, **kwargs):
    """
    Returns the ONNX graph which computes
    ``cdist(X, metric='sqeuclidean')``.
    """
    diff = OnnxSub('next_in', 'next', output_names=['diff'])
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1])
    flat = OnnxSqueeze(norm, output_names=['scan_out'], axes=[1])
    tensor_type = FloatTensorType if dtype == np.float32 else DoubleTensorType
    id_next.set_onnx_name_prefix('cdistsqe')
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', tensor_type()),
                     ('next', tensor_type())]),
        outputs=[('next_out', tensor_type()),
                 ('scan_out', tensor_type())],
        other_outputs=[flat],
        dtype=dtype)

    node = OnnxScan(X, Y, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph)
    return OnnxTranspose(node[1], perm=[1, 0],
                         **kwargs)
