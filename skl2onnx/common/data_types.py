# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxtk.common.data_types import DataType, Int64Type, FloatType  # noqa
from onnxtk.common.data_types import StringType, TensorType, Int64TensorType  # noqa
from onnxtk.common.data_types import FloatTensorType, StringTensorType  # noqa
from onnxtk.common.data_types import DictionaryType, SequenceType  # noqa
from onnxtk.common.data_types import find_type_conversion  # noqa


def onnx_built_with_ml():
    """
    Tells if ONNX was built with flag ``ONNX-ML``.
    """
    seq = SequenceType(FloatTensorType())
    try:
        seq.to_onnx_type()
        return True
    except RuntimeError:
        return False
