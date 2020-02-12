# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy

from onnxconverter_common import apply_identity
from onnx import TensorProto

from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSub, OnnxPow, OnnxDiv, OnnxMul, OnnxCast, OnnxNot, OnnxLess, OnnxLog, OnnxNeg, OnnxImputer
)


def convert_powertransformer(scope, operator, container):
    """Converter for PowerTransformer"""
    op_in = operator.inputs[0]
    op_out = operator.outputs[0].full_name
    op = operator.raw_operator
    lambdas = op.lambdas_

    # tensors of units and zeros
    ones_ = OnnxDiv(op_in, op_in)
    zeros_ = OnnxSub(op_in, op_in)

    # logical masks for input
    less_than_zero = OnnxLess(op_in, zeros_)
    less_mask = OnnxCast(less_than_zero, to=getattr(TensorProto, 'FLOAT'))

    greater_than_zero = OnnxNot(less_than_zero)
    greater_mask = OnnxCast(greater_than_zero, to=getattr(TensorProto, 'FLOAT'))

    # logical masks for lambdas
    lambda_zero_mask = OnnxCast((lambdas == 0), to=getattr(TensorProto, 'FLOAT'))
    lambda_nonzero_mask = OnnxCast((lambdas != 0), to=getattr(TensorProto, 'FLOAT'))
    lambda_two_mask = OnnxCast((lambdas == 2), to=getattr(TensorProto, 'FLOAT'))
    lambda_nontwo_mask = OnnxCast((lambdas != 2), to=getattr(TensorProto, 'FLOAT'))

    if 'yeo-johnson' in op.method:
        y0 = OnnxAdd(op_in, ones_)  # For positive input
        y1 = OnnxSub(ones_, op_in)  # For negative input

        # positive input, lambda != 0
        y_gr0_l_ne0 = OnnxPow(y0, lambdas)
        y_gr0_l_ne0 = OnnxSub(y_gr0_l_ne0, ones_)
        y_gr0_l_ne0 = OnnxDiv(y_gr0_l_ne0, lambdas)
        # y_gr0_l_ne0 = OnnxImputer(y_gr0_l_ne0, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        y_gr0_l_ne0 = OnnxImputer(y_gr0_l_ne0, imputed_value_floats=[0.0], replaced_value_float=numpy.inf)
        y_gr0_l_ne0 = OnnxMul(y_gr0_l_ne0, lambda_nonzero_mask)

        # positive input, lambda == 0
        y_gr0_l_eq0 = OnnxLog(y0)
        # y_gr0_l_eq0 = OnnxImputer(y_gr0_l_eq0, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        y_gr0_l_eq0 = OnnxMul(y_gr0_l_eq0, lambda_zero_mask)

        # positive input, an arbitrary lambda
        y_gr0 = OnnxAdd(y_gr0_l_ne0, y_gr0_l_eq0)
        y_gr0 = OnnxImputer(y_gr0, imputed_value_floats=[0.0],  replaced_value_float=numpy.NAN)
        y_gr0 = OnnxMul(y_gr0, greater_mask)

        # negative input, lambda != 2
        y_le0_l_ne2 = OnnxPow(y1, 2-lambdas)
        y_le0_l_ne2 = OnnxSub(ones_, y_le0_l_ne2)
        y_le0_l_ne2 = OnnxDiv(y_le0_l_ne2, 2-lambdas)
        # y_le0_l_ne2 = OnnxImputer(y_le0_l_ne2, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        y_le0_l_ne2 = OnnxImputer(y_le0_l_ne2, imputed_value_floats=[0.0], replaced_value_float=numpy.inf)
        y_le0_l_ne2 = OnnxMul(y_le0_l_ne2, lambda_nontwo_mask)

        # negative input, lambda == 2
        y_le0_l_eq2 = OnnxNeg(OnnxLog(y1))
        # y_le0_l_eq2 = OnnxImputer(y_le0_l_eq2, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        # y_le0_l_eq2 = OnnxImputer(y_le0_l_eq2, imputed_value_floats=[0.0], replaced_value_float=numpy.inf)
        y_le0_l_eq2 = OnnxMul(y_le0_l_eq2, lambda_two_mask)

        # negative input, an arbitrary lambda
        y_le0 = OnnxAdd(y_le0_l_ne2, y_le0_l_eq2)
        y_le0 = OnnxImputer(y_le0, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        y_le0 = OnnxMul(y_le0, less_mask)

        # Arbitrary input and lambda
        y = OnnxAdd(y_gr0, y_le0, output_names='tmp')

    elif 'box-cox' in op.method:
        # positive input, lambda != 0
        y_gr0_l_ne0 = OnnxPow(op_in, lambdas)
        y_gr0_l_ne0 = OnnxSub(y_gr0_l_ne0, ones_)
        y_gr0_l_ne0 = OnnxDiv(y_gr0_l_ne0, lambdas)
        # y_gr0_l_ne0 = OnnxImputer(y_gr0_l_ne0, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        y_gr0_l_ne0 = OnnxImputer(y_gr0_l_ne0, imputed_value_floats=[0.0], replaced_value_float=numpy.inf)
        y_gr0_l_ne0 = OnnxMul(y_gr0_l_ne0, lambda_nonzero_mask)

        # positive input, lambda == 0
        y_gr0_l_eq0 = OnnxLog(op_in)
        y_gr0_l_eq0 = OnnxImputer(y_gr0_l_eq0, imputed_value_floats=[0.0], replaced_value_float=numpy.NAN)
        y_gr0_l_eq0 = OnnxMul(y_gr0_l_eq0, lambda_zero_mask)

        # positive input, arbitrary lambda
        y = OnnxAdd(y_gr0_l_ne0, y_gr0_l_eq0, output_names='tmp')

        # negative input
        # model=PowerTransformer(method='box-cox').fit(negative_data) raises ValueError.
        # Therefore we cannot use convert_sklearn() for that model
    else:
        raise NotImplementedError('Method {} is not supported'.format(op.method))

    y.set_onnx_name_prefix('pref')
    y.add_to(scope, container)

    if op.standardize:
        name = scope.get_unique_operator_name('Scaler')
        attrs = dict(name=name,
                     offset=op._scaler.mean_,
                     scale=1.0 / op._scaler.scale_)
        container.add_node('Scaler', 'tmp', op_out, op_domain='ai.onnx.ml', **attrs)
    else:
        apply_identity(scope, 'tmp', op_out, container)


register_converter('SklearnPowerTransformer', convert_powertransformer)
