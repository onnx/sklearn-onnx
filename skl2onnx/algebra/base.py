# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from sklearn.base import BaseEstimator
from ..common._registration import get_converter, get_shape_calculator
from .._supported_operators import sklearn_operator_name_map
from .onnx_operator import OnnxOperator
from .type_helper import _guess_type


class OnnxOperatorMixin:
    """
    Base class for *scikit-learn* operators
    sharing an API to convert object to *ONNX*.
    """

    def to_onnx(self, X=None, name=None):
        """
        Converts the model in *ONNX* format.
        It calls method *_to_onnx* which must be
        overloaded.

        :param X: training data, at least one sample,
            it is used to guess the type of the input data.
        :param name: name of the model, if None,
            it is replaced by the the class name.
        """
        from .. import convert_sklearn
        if name is None:
            name = self.__class__.__name__
        if X is None:
            initial_types = self.infer_initial_type()
        else:
            gt = _guess_type(X)
            initial_types = [('X', gt)]
        return convert_sklearn(self, initial_types=initial_types)

    def infer_initial_types(self):
        """
        Infers initial types.
        """
        if hasattr(self, 'enumerate_initial_types'):
            return list(self.enumerate_initial_types())
        raise RuntimeError("Method enumerate_initial_types is missing "
                           "and initial_types are not defined")

    def _find_sklearn_parent(self):
        if hasattr(self.__class__, 'predict') and \
                "predict" in self.__class__.__dict__:
            raise RuntimeError("Method predict was modified. "
                               "There is no parser or converter available "
                               "for class '{}'.".format(self.__class__))
        if hasattr(self.__class__, 'transform') and \
                "transform" in self.__class__.__dict__:
            raise RuntimeError("Method transform was modified. "
                               "There is no parser or converter available "
                               "for class '{}'.".format(self.__class__))
        for cl in self.__class__.__bases__:
            if issubclass(cl, BaseEstimator):
                return cl
        raise RuntimeError("Unable to find any parent inherited from "
                           "BaseEstimator: {}.".format(
                               ", ".join(map(str, self.__class__.__bases__))))

    def to_onnx_operator(self, inputs=None, outputs=None):
        """
        This function must be overloaded.
        """
        raise NotImplementedError()

    def onnx_parser(self):
        """
        Returns a parser for this model.
        If not overloaded, it fetches the parser
        mapped to the first *scikit-learn* parent
        it can find.
        """
        try:
            op = self.to_onnx_operator()
        except NotImplementedError:
            self._find_sklearn_parent()
            return None

        def parser():
            names = []
            while True:
                try:
                    name = op.get_output(len(names))
                    if name is None:
                        break
                    names.append(name)
                except IndexError:
                    break
            return names
        return parser

    def get_inputs(self, inputs, i):
        if i >= len(inputs):
            return OnnxOperator.OnnxOperatorVariable(i)
        else:
            input = inputs[i]
            if isinstance(input, (str, OnnxOperator.UnscopedVariable)):
                return OnnxOperator.OnnxOperatorVariable(i, input)
            else:
                return input

    def onnx_converter(self):
        """
        Returns a converter for this model.
        If not overloaded, it fetches the converter
        mapped to the first *scikit-learn* parent
        it can find.
        """
        try:
            op = self.to_onnx_operator()
        except NotImplementedError:
            parent = self._find_sklearn_parent()
            name = sklearn_operator_name_map[parent]
            return get_converter(name)

        def converter(scope, operator, container):
            op.add_to(scope, container, operator=operator)

        return converter

    def onnx_shape_calculator(self):
        """
        Returns a shape calculator for this model.
        If not overloaded, it fetches the parser
        mapped to the first *scikit-learn* parent
        it can find.
        """
        try:
            op = self.to_onnx_operator()
        except NotImplementedError:
            parent = self._find_sklearn_parent()
            name = sklearn_operator_name_map[parent]
            return get_shape_calculator(name)

        def shape_calculator(operator):
            # onx = op.to_onnx(operator.inputs, operator.outputs)
            onames = [o.full_name for o in operator.outputs]
            return onames
        return shape_calculator
