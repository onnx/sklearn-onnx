# SPDX-License-Identifier: Apache-2.0


# Rather than using ONNX protobuf definition throughout our codebase,
# we import ONNX protobuf definition here so that we can conduct quick
# fixes by overwriting ONNX functions without changing any lines
# elsewhere.
from onnx import onnx_pb as onnx_proto
from onnx import defs

# Overwrite the make_tensor defined in onnx.helper because of a bug
# (string tensor get assigned twice)
from onnx.onnx_pb import TensorProto, ValueInfoProto

try:  # noqa: SIM105
    from onnx.onnx_pb import SparseTensorProto
except ImportError:
    # onnx is too old.
    pass


def get_opset_number_from_onnx():
    """
    Returns the latest opset version supported
    by the *onnx* package.
    """
    return defs.onnx_opset_version()


def get_latest_tested_opset_version():
    """
    This module relies on *onnxruntime* to test every
    converter. The function returns the most recent
    target opset tested with *onnxruntime* or the opset
    version specified by *onnx* package if this one is lower
    (return by `onnx.defs.onnx_opset_version()`).
    """
    from .. import __max_supported_opset__

    return min(__max_supported_opset__, get_opset_number_from_onnx())
