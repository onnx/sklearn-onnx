# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxtk.common.data_types import DataType, Int64Type, FloatType, StringType
from onnxtk.common.data_types import TensorType, Int64TensorType
from onnxtk.common.data_types import FloatTensorType, StringTensorType
from onnxtk.common.data_types import DictionaryType, SequenceType
from onnxtk.common.data_types import find_type_conversion

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
