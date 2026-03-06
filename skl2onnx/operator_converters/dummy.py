# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_proto_type
from ..common.utils_classifier import _finalize_converter_classes
from ..proto import onnx_proto
from ..algebra.onnx_ops import OnnxReshape, OnnxShape, OnnxSlice, OnnxTile


def convert_sklearn_dummy_regressor(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *DummyRegressor* into *ONNX* format.
    The model always outputs the precomputed constant value.
    """
    op_version = container.target_opset
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    dtype = {
        onnx_proto.TensorProto.DOUBLE: np.float64,
        onnx_proto.TensorProto.FLOAT: np.float32,
    }[proto_dtype]

    op = operator.raw_operator
    # constant_ has shape [1, n_outputs]; ravel to [n_outputs]
    fitted_constant = op.constant_.astype(dtype).ravel()
    n_outputs = op.n_outputs_

    # Get number of input rows (N) dynamically
    shape = OnnxShape(operator.inputs[0].full_name, op_version=op_version)
    first = OnnxSlice(
        shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        op_version=op_version,
    )

    # Tile constant n_outputs-element vector N times → shape [N * n_outputs]
    tiled = OnnxTile(fitted_constant, first, op_version=op_version)
    # Reshape to [N, n_outputs]
    final = OnnxReshape(
        tiled,
        np.array([-1, n_outputs], dtype=np.int64),
        op_version=op_version,
        output_names=operator.outputs[:1],
    )
    final.add_to(scope, container)


def convert_sklearn_dummy_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converts a *DummyClassifier* into *ONNX* format.

    Supported strategies: ``prior``, ``most_frequent``, ``constant``.
    Stochastic strategies (``stratified``, ``uniform``) are not supported
    because their output is random and cannot be replicated deterministically
    in ONNX.
    """
    op_version = container.target_opset
    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT
    dtype = {
        onnx_proto.TensorProto.DOUBLE: np.float64,
        onnx_proto.TensorProto.FLOAT: np.float32,
    }[proto_dtype]

    op = operator.raw_operator

    if op.n_outputs_ > 1:
        raise NotImplementedError(
            "DummyClassifier converter does not support multi-output "
            "classification (n_outputs_=%d)." % op.n_outputs_
        )

    strategy = op.strategy
    if strategy in ("stratified", "uniform"):
        raise NotImplementedError(
            "DummyClassifier with strategy=%r produces random predictions "
            "which cannot be represented deterministically in ONNX." % strategy
        )

    classes = op.classes_
    class_prior = op.class_prior_.astype(dtype)
    n_classes = len(classes)

    # Compute the fixed probability row and predicted class index
    if strategy == "prior":
        proba_row = class_prior
        predicted_idx = int(np.argmax(class_prior))
    elif strategy == "most_frequent":
        predicted_idx = int(np.argmax(class_prior))
        proba_row = np.zeros(n_classes, dtype=dtype)
        proba_row[predicted_idx] = dtype(1.0)
    elif strategy == "constant":
        constant_val = op.constant
        # Find the index of the constant value in classes
        matches = np.where(classes == constant_val)[0]
        if len(matches) == 0:
            raise ValueError(
                "DummyClassifier constant value %r not found in classes %r."
                % (constant_val, classes)
            )
        predicted_idx = int(matches[0])
        proba_row = np.zeros(n_classes, dtype=dtype)
        proba_row[predicted_idx] = dtype(1.0)
    else:
        raise NotImplementedError(
            "DummyClassifier with strategy=%r is not supported." % strategy
        )

    # Get number of input rows (N) dynamically
    shape = OnnxShape(operator.inputs[0].full_name, op_version=op_version)
    first = OnnxSlice(
        shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        op_version=op_version,
    )

    # --- Probability output [N, n_classes] ---
    # Tile the 1D proba_row N times → [N * n_classes], then reshape to [N, n_classes]
    proba_tiled = OnnxTile(proba_row, first, op_version=op_version)
    proba_final = OnnxReshape(
        proba_tiled,
        np.array([-1, n_classes], dtype=np.int64),
        op_version=op_version,
        output_names=operator.outputs[1:2],
    )
    proba_final.add_to(scope, container)

    # --- Label output [N] ---
    # Tile the predicted class index N times to get int64 indices, then
    # use _finalize_converter_classes to map indices → actual class labels
    idx_array = np.array([predicted_idx], dtype=np.int64)
    idx_tiled_name = scope.get_unique_variable_name("predicted_idx_tiled")
    idx_tiled = OnnxTile(
        idx_array, first, op_version=op_version, output_names=[idx_tiled_name]
    )
    idx_tiled.add_to(scope, container)

    _finalize_converter_classes(
        scope,
        idx_tiled_name,
        operator.outputs[0].full_name,
        container,
        classes,
        proto_dtype,
    )


register_converter("SklearnDummyRegressor", convert_sklearn_dummy_regressor)
register_converter(
    "SklearnDummyClassifier",
    convert_sklearn_dummy_classifier,
    options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
)
