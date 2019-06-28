# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from collections import OrderedDict
from ..common.data_types import FloatTensorType
from .onnx_ops import (
    OnnxIdentity, OnnxScan,
    OnnxSub, OnnxReduceSumSquare, OnnxSqueeze
)


def squareform_cdist(X, metric='sqeuclidean', **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(cdist(X, metric=metric)``.
    """
    if metric == 'sqeuclidean':
        return _squareform_cdist_sqeuclidean(X)
    else:
        raise NotImplementedError("metric='{}' is not implemented.".format(
            metric))


def _squareform_cdist_sqeuclidean(X, **kwargs):
    """
    Returns the ONNX graph which computes
    ``squareform(cdist(X, metric='sqeuclidean')``.
    """
    diff = OnnxSub('next_in', 'next', output_names=['diff'])
    id_next = OnnxIdentity('next_in', output_names=['next_out'])
    norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1])
    flat = OnnxSqueeze(norm, output_names=['scan_out'], axes=[1])
    scan_body = id_next.to_onnx(
        OrderedDict([('next_in', FloatTensorType()),
                     ('next', FloatTensorType())]),
        outputs=[('next_out', FloatTensorType([3, 2])),
                 ('scan_out', FloatTensorType([3]))],
        other_outputs=[flat])

    node = OnnxScan(X, X, output_names=['scan0_{idself}', 'scan1_{idself}'],
                    num_scan_inputs=1, body=scan_body.graph,
                    **kwargs)
    return node[1]
