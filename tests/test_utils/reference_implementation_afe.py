# SPDX-License-Identifier: Apache-2.0
"""
Helpers to test runtimes.
"""

from onnx.defs import onnx_opset_version


def _array_feature_extrator(data, indices):
    """
    Implementation of operator *ArrayFeatureExtractor*
    with :epkg:`numpy`.
    """
    if len(indices.shape) == 2 and indices.shape[0] == 1:
        index = indices.ravel().tolist()
        add = len(index)
    elif len(indices.shape) == 1:
        index = indices.tolist()
        add = len(index)
    else:
        add = 1
        for s in indices.shape:
            add *= s
        index = indices.ravel().tolist()
    new_shape = (1, add) if len(data.shape) == 1 else list(data.shape[:-1]) + [add]
    try:
        tem = data[..., index]
    except IndexError as e:
        raise RuntimeError(f"data.shape={data.shape}, indices={indices}") from e
    res = tem.reshape(new_shape)
    return res


if onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun

    class ArrayFeatureExtractor(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(self, data, indices):
            """
            Runtime for operator *ArrayFeatureExtractor*.

            .. warning::
                ONNX specifications may be imprecise in some cases.
                When the input data is a vector (one dimension),
                the output has still two like a matrix with one row.
                The implementation follows what :epkg:`onnxruntime` does in
                `array_feature_extractor.cc
                <https://github.com/microsoft/onnxruntime/blob/main/
                onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc#L84>`_.
            """
            res = _array_feature_extrator(data, indices)
            return (res,)
