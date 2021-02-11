# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import numpy as np
import onnx
from skl2onnx import __max_supported_opset__ as max_opset
from skl2onnx.common._topology import OPSET_TO_IR_VERSION
from .tests_helper import dump_data_and_model  # noqa
from .tests_helper import (  # noqa
    dump_one_class_classification,
    dump_binary_classification,
    dump_multilabel_classification,
    dump_multiple_classification,
)
from .tests_helper import (  # noqa
    dump_multiple_regression,
    dump_single_regression,
    convert_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    fit_regression_model,
    binary_array_to_string
)


def create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    elif H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    else:
        raise ValueError('This function only produce 2-D or 4-D tensor.')


def _get_ir_version(opv):
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


TARGET_OPSET = int(
    os.environ.get(
        'TEST_TARGET_OPSET',
        min(max_opset,
            onnx.defs.onnx_opset_version())))

TARGET_IR = int(
    os.environ.get(
        'TEST_TARGET_IR',
        min(OPSET_TO_IR_VERSION[TARGET_OPSET],
            _get_ir_version(TARGET_OPSET))))
