# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_numpy_type


def convert_quantile_transformer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """Converter for QuantileTransformer"""
    # op_in = operator.inputs[0]
    # op_out = operator.outputs[0].full_name
    op = operator.raw_operator
    # opv = container.target_opset
    dtype = guess_numpy_type(operator.inputs[0].type)
    if dtype != np.float64:
        dtype = np.float32
    if op.output_distribution != "uniform":
        raise RuntimeError(
            "Conversion of QuantileTransformer with output_distribution=%r "
            "is not supported." % op.output_distribution
        )

    # ref = op.references_
    # quantiles = op.quantiles_

    # Code of QuantileTransformer.transform
    # lower_bound_x = quantiles[0]
    # upper_bound_x = quantiles[-1]
    # lower_bound_y = 0
    # upper_bound_y = 1
    # lower_bounds_idx = (X_col == lower_bound_x)
    # upper_bounds_idx = (X_col == upper_bound_x)

    # isfinite_mask = ~np.isnan(X_col)
    # xcolf = X_col[isfinite_mask]
    # X_col[isfinite_mask] = .5 * (
    #     np.interp(xcolf, quantiles, self.references_)
    #     - np.interp(-xcolf, -quantiles[::-1], -self.references_[::-1]))
    # X_col[upper_bounds_idx] = upper_bound_y
    # X_col[lower_bounds_idx] = lower_bound_y

    # Strategy
    # implement interpolation in Onnx
    # * use 2 trees to determine the quantile x (qx, dx)
    # * use 2 trees to determine the quantile y (qy, dy)
    # do : (x - q) * dx * dy + qy

    # y.set_onnx_name_prefix('quantile')
    # y.add_to(scope, container)
    raise NotImplementedError()


register_converter("SklearnQuantileTransformer", convert_quantile_transformer)
