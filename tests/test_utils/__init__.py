# SPDX-License-Identifier: Apache-2.0

import os
from distutils.version import StrictVersion
import numpy as np
import onnx
from onnxruntime import __version__ as ort_version
from skl2onnx import __max_supported_opset__ as max_opset
from skl2onnx.common._topology import OPSET_TO_IR_VERSION
from .tests_helper import dump_data_and_model  # noqa
from .tests_helper import (  # noqa
    dump_one_class_classification,
    dump_binary_classification,
    dump_multilabel_classification,
    dump_multiple_classification)
from .tests_helper import (  # noqa
    dump_multiple_regression,
    dump_single_regression,
    convert_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    fit_clustering_model,
    fit_regression_model,
    binary_array_to_string,
    path_to_leaf
)


def create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    elif H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    else:
        raise ValueError('This function only produce 2-D or 4-D tensor.')


def _get_ir_version(opv):
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
    <https://github.com/microsoft/onnxruntime/blob/
    master/docs/Versioning.md>`_.
    """
    vi = StrictVersion(ort_version.split('+')[0])
    if vi >= StrictVersion("1.9.0"):
        return 15
    if vi >= StrictVersion("1.8.0"):
        return 14
    if vi >= StrictVersion("1.6.0"):
        return 13
    if vi >= StrictVersion("1.3.0"):
        return 12
    if vi >= StrictVersion("1.0.0"):
        return 11
    if vi >= StrictVersion("0.4.0"):
        return 10
    if vi >= StrictVersion("0.3.0"):
        return 9
    return 8


TARGET_OPSET = int(
    os.environ.get(
        'TEST_TARGET_OPSET',
        min(max_onnxruntime_opset(),
            min(max_opset,
                onnx.defs.onnx_opset_version()))))

TARGET_IR = int(
    os.environ.get(
        'TEST_TARGET_IR',
        min(OPSET_TO_IR_VERSION[TARGET_OPSET],
            _get_ir_version(TARGET_OPSET))))
