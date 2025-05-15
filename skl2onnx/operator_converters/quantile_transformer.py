# SPDX-License-Identifier: Apache-2.0

import numpy as np
import math
from onnx import TensorProto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_numpy_type, guess_proto_type
from ..algebra.onnx_ops import (
    OnnxWhere,
    OnnxIsNaN,
    OnnxNot,
    OnnxCast,
    OnnxEqual,
    OnnxNeg,
    OnnxSub,
    OnnxMul,
    OnnxAdd,
    OnnxDiv,
    OnnxGather,
    OnnxLessOrEqual,
    OnnxReshape,
    OnnxShape,
    OnnxFlatten,
    OnnxGreaterOrEqual,
    OnnxOr,
)


def _next_power_of_two(n):
    return 1 << (n).bit_length()


def _pad_quantiles(quantiles, target_size, n_quantiles, mode="min"):
    pad_before = target_size - n_quantiles + 1
    if mode == "min":
        pad_values = np.repeat(quantiles[0:1, :], pad_before, axis=0)
    else:
        pad_values = np.repeat(quantiles[-1:, :], pad_before, axis=0)
    return np.concatenate([pad_values, quantiles], axis=0)


def _pad_references(references, target_size, n_quantiles, mode="min"):
    pad_before = target_size - n_quantiles + 1
    if mode == "min":
        pad_values = np.repeat([references[0]], pad_before)
    else:
        pad_values = np.repeat([references[-1]], pad_before)
    return np.concatenate([pad_values, references])


def convert_quantile_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op = operator.raw_operator
    op_in = operator.inputs[0]
    op_out = operator.outputs[0].full_name
    opv = container.target_opset

    if op.output_distribution != "uniform":
        raise RuntimeError(
            "Conversion of QuantileTransformer with output_distribution=%r "
            "is not supported." % op.output_distribution
        )

    dtype = guess_numpy_type(op_in.type)
    if dtype != np.float64:
        dtype = np.float32
    proto_dtype = guess_proto_type(op_in.type)

    quantiles = op.quantiles_.astype(dtype)
    references = op.references_.astype(dtype)
    n_quantiles, n_features = quantiles.shape

    target_size = _next_power_of_two(n_quantiles)

    # Correct padding with min/max values
    quantiles_pad = _pad_quantiles(quantiles, target_size, n_quantiles, "min")
    references_pad = _pad_references(references, target_size, n_quantiles, "min")

    # Prepare reversed and negated quantiles and references
    quantiles_neg_rev = -quantiles[::-1, :]
    references_neg_rev = -references[::-1]

    quantiles_neg_rev_pad = _pad_quantiles(
        quantiles_neg_rev, target_size, n_quantiles, "min"
    )
    references_neg_rev_pad = _pad_references(
        references_neg_rev, target_size, n_quantiles, "min"
    )

    # Prepare initializers
    def prepare_initializer(arr, name, transpose=False):
        if transpose:
            arr = arr.T.flatten()
        else:
            arr = arr.flatten()
        arr_name = scope.get_unique_variable_name(name)
        container.add_initializer(arr_name, proto_dtype, arr.shape, arr)
        return arr_name

    quantiles_name = prepare_initializer(quantiles_pad, "quantiles", transpose=True)
    references_name = prepare_initializer(
        np.tile(references_pad, n_features), "references"
    )
    quantiles_neg_rev_name = prepare_initializer(
        quantiles_neg_rev_pad, "quantiles_neg_rev", transpose=True
    )
    references_neg_rev_name = prepare_initializer(
        np.tile(references_neg_rev_pad, n_features), "references_neg_rev"
    )

    is_nan = OnnxIsNaN(op_in, op_version=opv)
    is_finite = OnnxNot(is_nan, op_version=opv)
    X_clean = OnnxWhere(is_finite, op_in, op_in, op_version=opv)
    X_clean.add_to(scope, container)

    def create_interpolation(X_input, quantiles_init, references_init):
        flat_input = OnnxFlatten(X_input, axis=0, op_version=opv)
        flat_input.add_to(scope, container)

        current_index = OnnxMul(
            OnnxCast(X_input, to=np.int64, op_version=opv),
            np.array(0, dtype=np.int64),
            op_version=opv,
        )
        current_index = OnnxFlatten(
            OnnxAdd(
                current_index,
                np.array(
                    [i * (target_size + 1) for i in range(n_features)], dtype=np.int64
                ),
                op_version=opv,
            ),
            axis=0,
            op_version=opv,
        )
        current_index.add_to(scope, container)

        step = OnnxFlatten(
            OnnxAdd(
                OnnxMul(current_index, np.array(0, dtype=np.int64), op_version=opv),
                np.array([target_size // 2], dtype=np.int64),
                op_version=opv,
            ),
            op_version=opv,
        )
        step.add_to(scope, container)

        for _i in range(int(math.log2(target_size))):
            current_index_temp = OnnxAdd(current_index, step, op_version=opv)
            current_index_temp.add_to(scope, container)

            q_current = OnnxGather(
                quantiles_init, current_index_temp, axis=0, op_version=opv
            )
            q_current.add_to(scope, container)

            mask = OnnxGreaterOrEqual(flat_input, q_current, op_version=opv)
            mask.add_to(scope, container)

            current_index = OnnxWhere(
                mask, current_index_temp, current_index, op_version=opv
            )
            current_index.add_to(scope, container)

            step = OnnxCast(
                OnnxDiv(step, np.array(2, dtype=np.int64), op_version=opv),
                to=TensorProto.INT64,
                op_version=opv,
            )
            step.add_to(scope, container)

        index_plus_one = OnnxAdd(
            current_index, np.array(1, dtype=np.int64), op_version=opv
        )
        index_plus_one.add_to(scope, container)

        q_low = OnnxGather(quantiles_init, current_index, axis=0, op_version=opv)
        q_high = OnnxGather(quantiles_init, index_plus_one, axis=0, op_version=opv)
        r_low = OnnxGather(references_init, current_index, axis=0, op_version=opv)
        r_high = OnnxGather(references_init, index_plus_one, axis=0, op_version=opv)

        # Handle boundary conditions
        mask_low = OnnxLessOrEqual(flat_input, q_low, op_version=opv)
        mask_high = OnnxGreaterOrEqual(flat_input, q_high, op_version=opv)

        delta_q = OnnxSub(q_high, q_low, op_version=opv)
        delta_r = OnnxSub(r_high, r_low, op_version=opv)

        slope = OnnxDiv(delta_r, delta_q, op_version=opv)
        invalid_slope = OnnxOr(
            OnnxLessOrEqual(delta_q, np.array(0, dtype=dtype), op_version=opv),
            OnnxIsNaN(slope, op_version=opv),
            op_version=opv,
        )
        slope = OnnxWhere(
            invalid_slope, np.array(0, dtype=dtype), slope, op_version=opv
        )

        interp_normal = OnnxAdd(
            r_low,
            OnnxMul(slope, OnnxSub(flat_input, q_low, op_version=opv), op_version=opv),
            op_version=opv,
        )

        interp = OnnxWhere(
            mask_high,
            r_high,
            OnnxWhere(mask_low, r_low, interp_normal, op_version=opv),
            op_version=opv,
        )
        return interp

    # First interpolation
    interp1 = create_interpolation(X_clean, quantiles_name, references_name)

    # Second interpolation
    X_neg = OnnxNeg(X_clean, op_version=opv)
    X_neg.add_to(scope, container)
    interp2 = create_interpolation(
        X_neg, quantiles_neg_rev_name, references_neg_rev_name
    )

    # Combine results
    final_interp = OnnxMul(
        OnnxSub(interp1, interp2, op_version=opv),
        np.array(0.5, dtype=dtype),
        op_version=opv,
    )

    # Reshape to original shape
    output_shape = OnnxShape(X_clean, op_version=opv)
    final = OnnxReshape(
        final_interp, output_shape, op_version=opv, output_names=[op_out]
    )
    min_quantiles = quantiles[0, :]  # Minimum values of quantiles
    max_quantiles = quantiles[-1, :]  # Maximum values of quantiles

    # Comparison with minimum and maximum values
    min_mask = OnnxFlatten(
        OnnxEqual(X_clean, min_quantiles, op_version=opv), axis=0, op_version=opv
    )
    max_mask = OnnxFlatten(
        OnnxEqual(X_clean, max_quantiles, op_version=opv), axis=0, op_version=opv
    )

    # Set 0 for minimum values and 1 for maximum values
    final = OnnxWhere(
        max_mask,
        np.array(1.0, dtype=dtype),
        OnnxWhere(
            min_mask,
            OnnxMul(final_interp, np.array(0.0, dtype=dtype), op_version=opv),
            final_interp,
            op_version=opv,
        ),
        op_version=opv,
    )
    final = OnnxReshape(final, output_shape, op_version=opv, output_names=[op_out])
    final.add_to(scope, container)


register_converter("SklearnQuantileTransformer", convert_quantile_transformer)
