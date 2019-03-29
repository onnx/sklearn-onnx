# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin
from ..common.data_types import FloatTensorType
from ..common._topology import Variable


class SymbolicOpTransformer(BaseEstimator, TransformerMixin):
    """
    Implements a bridge between ONNX operators
    and symbolic operators.

    :param op: :class:`OnnxOperator <skl2onnx.algebra.OnnxOperator>`
    """

    def __init__(self, op):
        self.op = op

    def __str__(self):
        return "Sym('%s')" % self.op.__class__.__name__

    def convert(self, scope, operator, container):
        """
        Converts the operator into an ONNX graph.
        """
        self.op.add_to(scope, operator, container)

    def set_shape(self, operator):
        """
        Returns the shapes of the outputs.
        """
        outputs = self.op.get_expected_outputs(operator.inputs)
        for i in range(0, len(outputs)):
            fr_out = outputs[i]
            to_out = operator.outputs[i]
            to_out.type = fr_out.type
            to_out.type.shape = fr_out.type.shape

    def parse(self, scope, model, inputs, custom_parsers=None):
        """
        Defines the number of expected inputs.
        """
        var_inputs = self.op.check_inputs(inputs)
        var_inputs = [Variable(n, n, scope, type=t) for n, t in var_inputs]
        name = self.op.__class__.__name__
        this_operator = scope.declare_local_operator(name, model)
        this_operator.inputs = inputs
        nb_out = len(self.op.get_expected_outputs(var_inputs))
        for i in range(nb_out):
            var = scope.declare_local_variable(name + "_o", FloatTensorType())
            this_operator.outputs.append(var)
        return this_operator.outputs
