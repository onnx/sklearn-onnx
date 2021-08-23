# SPDX-License-Identifier: Apache-2.0


from ..common._registration import register_shape_calculator

_stack = []


def multioutput_regressor_shape_calculator(operator):
    """Shape calculator for MultiOutputRegressor"""
    i = operator.inputs[0]
    o = operator.outputs[0]
    N = i.get_first_dimension()
    C = len(operator.raw_operator.estimators_)
    o.type = o.type.__class__([N, C])


register_shape_calculator('SklearnMultiOutputRegressor',
                          multioutput_regressor_shape_calculator)
