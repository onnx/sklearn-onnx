# SPDX-License-Identifier: Apache-2.0
"""
Converter for sklearn.preprocessing.SplineTransformer.

The B-spline design matrix is computed using the Cox-de Boor recursion formula:
  B[i, 0](x) = 1 if t[i] <= x < t[i+1], else 0
  B[i, j](x) = (x - t[i]) / (t[i+j] - t[i]) * B[i, j-1](x)
               + (t[i+j+1] - x) / (t[i+j+1] - t[i+1]) * B[i+1, j-1](x)

Extrapolation modes:
  - 'constant': out-of-range values use boundary spline values (precomputed)
  - 'linear':   out-of-range values use linear extrapolation from boundary
  - 'continue': polynomial pieces extended beyond boundary using clamped interval
  - 'periodic': input mapped to periodic range before evaluation
  - 'error':    treated same as 'continue' (ONNX cannot raise errors at runtime)
"""

import numpy as np
from onnx import TensorProto
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_numpy_type, guess_proto_type
from ..algebra.onnx_ops import (
    OnnxAdd,
    OnnxSub,
    OnnxMul,
    OnnxDiv,
    OnnxCast,
    OnnxNot,
    OnnxLess,
    OnnxLessOrEqual,
    OnnxAnd,
    OnnxWhere,
    OnnxGather,
    OnnxConcat,
    OnnxReshape,
    OnnxIdentity,
    OnnxFloor,
)


def _bspline_design_matrix_onnx(x_col, t, k, dtype, opv, extrapolate_boundary=False):
    """
    Build ONNX graph computing the B-spline design matrix for a single feature.

    Parameters
    ----------
    x_col : OnnxOp
        1-D input of shape [n_samples].
    t : np.ndarray
        Knot vector (already cast to *dtype*).
    k : int
        B-spline degree.
    dtype : numpy dtype
        Float dtype used throughout.
    opv : int
        ONNX opset version.
    extrapolate_boundary : bool
        If True, clamp the interval selection so that the polynomial piece at
        the boundary is extended (needed for ``extrapolation='continue'``).

    Returns
    -------
    OnnxOp
        Node of shape [n_samples, m - k - 1] where m = len(t).
    """
    m = len(t)
    proto_dtype = TensorProto.DOUBLE if dtype == np.float64 else TensorProto.FLOAT

    # Reshape x to [n_samples, 1] for broadcasting against knot constants.
    x = OnnxReshape(x_col, np.array([-1, 1], dtype=np.int64), op_version=opv)

    t_left = t[:-1].reshape(1, -1).astype(dtype)  # [1, m-1]
    t_right = t[1:].reshape(1, -1).astype(dtype)  # [1, m-1]

    # --- Degree-0 indicator functions ---
    # B[i,0](x) = 1 if t[i] <= x < t[i+1], else 0
    geq = OnnxNot(OnnxLess(x, t_left, op_version=opv), op_version=opv)
    lt = OnnxLess(x, t_right, op_version=opv)
    B_interior = OnnxCast(
        OnnxAnd(geq, lt, op_version=opv), to=proto_dtype, op_version=opv
    )

    if extrapolate_boundary:
        # For x < xmin: use first valid interval (index k).
        # For x >= xmax: use last valid interval (index m-k-2).
        xmin = t[k].reshape(1, 1).astype(dtype)
        xmax = t[-k - 1].reshape(1, 1).astype(dtype)

        b_left_data = np.zeros((1, m - 1), dtype=dtype)
        b_left_data[0, k] = 1.0
        b_right_data = np.zeros((1, m - 1), dtype=dtype)
        b_right_data[0, m - k - 2] = 1.0

        below = OnnxLess(x, xmin, op_version=opv)
        above = OnnxNot(OnnxLessOrEqual(x, xmax, op_version=opv), op_version=opv)

        B = OnnxWhere(
            below,
            b_left_data,
            OnnxWhere(above, b_right_data, B_interior, op_version=opv),
            op_version=opv,
        )
    else:
        B = B_interior

    # --- De Boor recursion ---
    for j in range(1, k + 1):
        n_curr = m - j - 1  # number of basis functions at this level

        # alpha[i] = (x - t[i]) / (t[i+j] - t[i]),  with 0/0 = 0
        ta_a = t[:n_curr].reshape(1, -1).astype(dtype)
        ta_b = t[j : j + n_curr].reshape(1, -1).astype(dtype)
        denom_a = (ta_b - ta_a).astype(dtype)
        safe_da = np.where(denom_a != 0, denom_a, dtype(1.0)).astype(dtype)
        nz_a = (denom_a != 0).astype(dtype)

        alpha = OnnxMul(
            OnnxDiv(OnnxSub(x, ta_a, op_version=opv), safe_da, op_version=opv),
            nz_a,
            op_version=opv,
        )

        # beta[i] = (t[i+j+1] - x) / (t[i+j+1] - t[i+1]),  with 0/0 = 0
        tb_a = t[1 : 1 + n_curr].reshape(1, -1).astype(dtype)
        tb_b = t[j + 1 : j + 1 + n_curr].reshape(1, -1).astype(dtype)
        denom_b = (tb_b - tb_a).astype(dtype)
        safe_db = np.where(denom_b != 0, denom_b, dtype(1.0)).astype(dtype)
        nz_b = (denom_b != 0).astype(dtype)

        beta = OnnxMul(
            OnnxDiv(OnnxSub(tb_b, x, op_version=opv), safe_db, op_version=opv),
            nz_b,
            op_version=opv,
        )

        # Slice B to B_left (cols 0..n_curr-1) and B_right (cols 1..n_curr).
        idx_left = np.arange(n_curr, dtype=np.int64)
        idx_right = np.arange(1, n_curr + 1, dtype=np.int64)
        B_left = OnnxGather(B, idx_left, axis=1, op_version=opv)
        B_right = OnnxGather(B, idx_right, axis=1, op_version=opv)

        B = OnnxAdd(
            OnnxMul(alpha, B_left, op_version=opv),
            OnnxMul(beta, B_right, op_version=opv),
            op_version=opv,
        )

    return B  # shape [n_samples, n_splines]


def convert_sklearn_spline_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """Converter for SplineTransformer."""
    op = operator.raw_operator
    op_in = operator.inputs[0]
    op_out = operator.outputs[0].full_name
    opv = container.target_opset

    dtype = guess_numpy_type(op_in.type)
    if dtype != np.float64:
        dtype = np.float32

    n_features = op.n_features_in_
    n_splines = op.bsplines_[0].c.shape[1]
    degree = op.degree
    extrapolation = op.extrapolation

    # Cast input once if necessary.
    if dtype == np.float32:
        X = OnnxCast(op_in, to=TensorProto.FLOAT, op_version=opv)
    else:
        X = OnnxCast(op_in, to=TensorProto.DOUBLE, op_version=opv)

    feature_outputs = []

    for f_idx in range(n_features):
        spl = op.bsplines_[f_idx]
        t = spl.t.astype(dtype)
        k = spl.k
        m = len(t)
        # Use original float64 knot values for spline evaluation to avoid
        # precision issues when the BSpline has extrapolate=False.
        xmin_f64 = spl.t[k]
        xmax_f64 = spl.t[-k - 1]
        xmin = dtype(xmin_f64)
        xmax = dtype(xmax_f64)

        # Extract feature column: shape [n_samples, 1] then flatten to [n_samples].
        col_2d = OnnxGather(X, np.array(f_idx, dtype=np.int64), axis=1, op_version=opv)
        x_col = OnnxReshape(
            col_2d, np.array([-1], dtype=np.int64), op_version=opv
        )

        if extrapolation == "periodic":
            n_valid = m - k - 1
            period = dtype(t[n_valid] - t[k])
            if period > 0:
                # x_mapped = xmin + (x - xmin) % period
                # Using: a % b = a - b * floor(a / b)
                x_shifted = OnnxSub(x_col, np.array([xmin], dtype=dtype), op_version=opv)
                x_div = OnnxDiv(x_shifted, np.array([period], dtype=dtype), op_version=opv)
                x_floor = OnnxFloor(x_div, op_version=opv)
                x_mod = OnnxSub(
                    x_shifted,
                    OnnxMul(np.array([period], dtype=dtype), x_floor, op_version=opv),
                    op_version=opv,
                )
                x_col = OnnxAdd(
                    np.array([xmin], dtype=dtype), x_mod, op_version=opv
                )
                x_col = OnnxReshape(
                    x_col, np.array([-1], dtype=np.int64), op_version=opv
                )
            else:
                x_col = OnnxMul(
                    x_col, np.array([dtype(0.0)], dtype=dtype), op_version=opv
                )

        # Compute B-spline design matrix for this feature.
        extrapolate_boundary = extrapolation == "continue"
        B = _bspline_design_matrix_onnx(x_col, t, k, dtype, opv, extrapolate_boundary)

        # --- Handle extrapolation modes ---
        if extrapolation == "periodic":
            # Adjust for periodic: B[:, :degree] += B[:, -degree:]
            # then drop the last `degree` columns.
            if degree > 0:
                n_b = n_splines + degree  # full (un-trimmed) columns
                idx_first = np.arange(degree, dtype=np.int64)
                idx_last = np.arange(n_b - degree, n_b, dtype=np.int64)
                B_first = OnnxGather(B, idx_first, axis=1, op_version=opv)
                B_last = OnnxGather(B, idx_last, axis=1, op_version=opv)
                B_first_adj = OnnxAdd(B_first, B_last, op_version=opv)
                idx_mid = np.arange(degree, n_b - degree, dtype=np.int64)
                B_mid = OnnxGather(B, idx_mid, axis=1, op_version=opv)
                B = OnnxConcat(B_first_adj, B_mid, axis=1, op_version=opv)

        elif extrapolation in ("constant", "error"):
            # Precompute boundary spline values using float64 knots to avoid
            # precision issues (BSpline may have extrapolate=False).
            f_min = spl(xmin_f64).astype(dtype)  # [n_splines]
            f_max = spl(xmax_f64).astype(dtype)  # [n_splines]

            # Reshape x_col to [n_samples, 1] for broadcasting.
            x_2d = OnnxReshape(
                x_col, np.array([-1, 1], dtype=np.int64), op_version=opv
            )
            below = OnnxLess(x_2d, np.array([[xmin]], dtype=dtype), op_version=opv)
            above = OnnxNot(
                OnnxLessOrEqual(x_2d, np.array([[xmax]], dtype=dtype), op_version=opv),
                op_version=opv,
            )
            B = OnnxWhere(
                below,
                f_min.reshape(1, -1),
                OnnxWhere(above, f_max.reshape(1, -1), B, op_version=opv),
                op_version=opv,
            )

        elif extrapolation == "linear":
            # Precompute boundary values and first derivatives using float64
            # knots to avoid precision issues (BSpline may have extrapolate=False).
            f_min = spl(xmin_f64).astype(dtype)
            f_max = spl(xmax_f64).astype(dtype)
            fp_min = spl(xmin_f64, nu=1).astype(dtype)
            fp_max = spl(xmax_f64, nu=1).astype(dtype)

            # For degree <= 1 sklearn widens the extrapolation range by 1.
            deg = degree if degree > 1 else degree + 1

            x_2d = OnnxReshape(
                x_col, np.array([-1, 1], dtype=np.int64), op_version=opv
            )
            below = OnnxLess(x_2d, np.array([[xmin]], dtype=dtype), op_version=opv)
            above = OnnxNot(
                OnnxLessOrEqual(x_2d, np.array([[xmax]], dtype=dtype), op_version=opv),
                op_version=opv,
            )

            # Build masked coefficient vectors (zero-padded to n_splines).
            # For below: first `deg` entries from f_min/fp_min, rest 0.
            f_min_masked = np.zeros(n_splines, dtype=dtype)
            f_min_masked[:deg] = f_min[:deg]
            fp_min_masked = np.zeros(n_splines, dtype=dtype)
            fp_min_masked[:deg] = fp_min[:deg]

            # For above: last `deg` entries from f_max/fp_max, rest 0.
            f_max_masked = np.zeros(n_splines, dtype=dtype)
            f_max_masked[n_splines - deg :] = f_max[n_splines - deg :]
            fp_max_masked = np.zeros(n_splines, dtype=dtype)
            fp_max_masked[n_splines - deg :] = fp_max[n_splines - deg :]

            # extrap_below[j] = f_min_masked[j] + (x - xmin) * fp_min_masked[j]
            # Broadcasting: [n_samples, 1] * [1, n_splines] -> [n_samples, n_splines]
            dx_below = OnnxSub(
                x_2d, np.array([[xmin]], dtype=dtype), op_version=opv
            )
            extrap_below = OnnxAdd(
                f_min_masked.reshape(1, -1),
                OnnxMul(dx_below, fp_min_masked.reshape(1, -1), op_version=opv),
                op_version=opv,
            )

            # extrap_above[j] = f_max_masked[j] + (x - xmax) * fp_max_masked[j]
            dx_above = OnnxSub(
                x_2d, np.array([[xmax]], dtype=dtype), op_version=opv
            )
            extrap_above = OnnxAdd(
                f_max_masked.reshape(1, -1),
                OnnxMul(dx_above, fp_max_masked.reshape(1, -1), op_version=opv),
                op_version=opv,
            )

            B = OnnxWhere(
                below,
                extrap_below,
                OnnxWhere(above, extrap_above, B, op_version=opv),
                op_version=opv,
            )

        feature_outputs.append(B)

    # Concatenate all feature outputs along axis 1.
    if len(feature_outputs) == 1:
        result = feature_outputs[0]
    else:
        result = OnnxConcat(*feature_outputs, axis=1, op_version=opv)

    # Handle include_bias=False: drop the last basis function for each feature.
    if not op.include_bias:
        n_total = n_features * n_splines
        keep = np.array(
            [j for j in range(n_total) if (j + 1) % n_splines != 0], dtype=np.int64
        )
        result = OnnxGather(result, keep, axis=1, op_version=opv)

    final = OnnxIdentity(result, op_version=opv, output_names=[op_out])
    final.add_to(scope, container)


register_converter("SklearnSplineTransformer", convert_sklearn_spline_transformer)
