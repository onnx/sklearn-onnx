# SPDX-License-Identifier: Apache-2.0

import os
import packaging.version as pv
import numpy as np
import onnx
from onnxruntime import __version__ as ort_version
from skl2onnx import __max_supported_opset__ as max_opset
from skl2onnx.common._topology import OPSET_TO_IR_VERSION
from .tests_helper import dump_data_and_model
from .tests_helper import (
    dump_one_class_classification,
    dump_binary_classification,
    dump_multilabel_classification,
    dump_multiple_classification,
)
from .tests_helper import (
    dump_multiple_regression,
    dump_single_regression,
    convert_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    fit_multi_output_classification_model,
    fit_clustering_model,
    fit_regression_model,
    binary_array_to_string,
    path_to_leaf,
)

try:
    from .utils_backend_onnx import ReferenceEvaluatorEx
except ImportError:

    def ReferenceEvaluatorEx(*args, **kwargs):
        raise NotImplementedError(
            "onnx package does not implement class ReferenceEvaluator. "
            "Update to onnx>=1.13.0."
        )


def InferenceSessionEx(onx, *args, verbose=0, **kwargs):
    from onnxruntime import InferenceSession

    if "providers" not in kwargs:
        kwargs["providers"] = ["CPUExecutionProvider"]
    try:
        return InferenceSession(onx, *args, **kwargs)
    except Exception as e:
        if TARGET_OPSET >= 18 and "support for domain ai.onnx is till opset" in str(e):
            return ReferenceEvaluatorEx(onx, verbose=verbose)
        raise e


def create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    if H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    raise ValueError("This function only produce 2-D or 4-D tensor.")


def _get_ir_version(opv):
    if opv >= 21:
        return 10
    if opv >= 19:
        return 9
    if opv >= 15:
        return 8
    if opv >= 12:
        return 7
    if opv >= 11:
        return 6
    if opv >= 10:
        return 5
    if opv >= 9:
        return 4
    if opv >= 8:
        return 4
    return 3


def max_onnxruntime_opset():
    """
    See `Versioning.md
    <https://github.com/microsoft/onnxruntime/blob/main/docs/Versioning.md>`_.
    """
    vi = pv.Version(ort_version.split("+")[0])
    if vi >= pv.Version("1.18.0"):
        return 21
    if vi >= pv.Version("1.17.0"):
        return 20
    if vi >= pv.Version("1.15.0"):
        return 19
    if vi >= pv.Version("1.14.0"):
        return 18
    if vi >= pv.Version("1.12.0"):
        return 17
    if vi >= pv.Version("1.11.0"):
        return 16
    if vi >= pv.Version("1.10.0"):
        return 15
    if vi >= pv.Version("1.9.0"):
        return 15
    if vi >= pv.Version("1.8.0"):
        return 14
    if vi >= pv.Version("1.6.0"):
        return 13
    if vi >= pv.Version("1.3.0"):
        return 12
    if vi >= pv.Version("1.0.0"):
        return 11
    if vi >= pv.Version("0.4.0"):
        return 10
    if vi >= pv.Version("0.3.0"):
        return 9
    return 8


TARGET_OPSET = int(
    os.environ.get(
        "TEST_TARGET_OPSET",
        min(
            max_onnxruntime_opset(),
            min(max_opset, onnx.defs.onnx_opset_version()),
        ),
    )
)

# opset-ml == 4 still not implemented in onnxruntime
value_ml = 3
if TARGET_OPSET <= 16:
    # TreeEnsemble* for opset-ml == 3 is implemented in onnxruntime==1.12.0
    # but not in onnxruntime==1.11.0.
    value_ml = 2
if TARGET_OPSET <= 11:
    value_ml = 1

TARGET_OPSET_ML = int(os.environ.get("TEST_TARGET_OPSET_ML", value_ml))

TARGET_IR = int(
    os.environ.get(
        "TEST_TARGET_IR",
        min(OPSET_TO_IR_VERSION[TARGET_OPSET], _get_ir_version(TARGET_OPSET)),
    )
)
