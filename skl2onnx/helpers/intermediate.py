# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import textwrap
import warnings
from types import MethodType
from .. import convert_sklearn
from ..helpers.onnx_helper import select_model_inputs_outputs


class BaseEstimatorDebugInformation:
    """
    Stores information when the outputs of a pipeline
    is computed. It as added by function
    :func:`_alter_model_for_debugging`.
    """

    def __init__(self, model):
        self.model = model
        self.inputs = {}
        self.outputs = {}
        self.methods = {}
        if hasattr(model, "transform") and callable(model.transform):
            model._debug_transform = model.transform
            self.methods["transform"] = \
                lambda model, X: model._debug_transform(X)
        if hasattr(model, "predict") and callable(model.predict):
            model._debug_predict = model.predict
            self.methods["predict"] = lambda model, X: model._debug_predict(X)
        if hasattr(model, "predict_proba") and callable(model.predict_proba):
            model._debug_predict_proba = model.predict_proba
            self.methods["predict_proba"] = \
                lambda model, X: model._predict_proba(X)
        if hasattr(model, "decision_function") and callable(
            model.decision_function):  # noqa
            model._debug_decision_function = model.decision_function  # noqa
            self.methods["decision_function"] = \
                lambda model, X: model._decision_function(X)

    def __repr__(self):
        """
        usual
        """
        return self.to_str()

    def to_str(self, nrows=5):
        """
        Tries to produce a readable message.
        """
        rows = ['BaseEstimatorDebugInformation({})'.format(
            self.model.__class__.__name__)]
        for k in sorted(self.inputs):
            if k in self.outputs:
                rows.append('  ' + k + '(')
                self.display(self.inputs[k], nrows)
                rows.append(textwrap.indent(
                    self.display(self.inputs[k], nrows), '   '))
                rows.append('  ) -> (')
                rows.append(textwrap.indent(
                    self.display(self.outputs[k], nrows), '   '))
                rows.append('  )')
            else:
                raise KeyError(
                    "Unable to find output for method '{}'.".format(k))
        return "\n".join(rows)

    def display(self, data, nrows):
        """
        Displays the first
        """
        text = str(data)
        rows = text.split('\n')
        if len(rows) > nrows:
            rows = rows[:nrows]
            rows.append('...')
        if hasattr(data, 'shape'):
            rows.insert(0, "shape={}".format(data.shape))
        return "\n".join(rows)


def _alter_model_for_debugging(skl_model):
    """
    Overwrite methods transform, predict or predict_proba
    to collect the last inputs and outputs
    seen in these methods.
    """

    def transform(self, X, *args, **kwargs):
        self._debug.inputs['transform'] = X
        y = self._debug.methods['transform'](self, X, *args, **kwargs)
        self._debug.outputs['transform'] = y
        return y

    def predict(self, X, *args, **kwargs):
        self._debug.inputs['predict'] = X
        y = self._debug.methods['predict'](self, X, *args, **kwargs)
        self._debug.outputs['predict'] = y
        return y

    def predict_proba(self, X, *args, **kwargs):
        self._debug.inputs['predict_proba'] = X
        y = self._debug.methods['predict_proba'](self, X, *args, **kwargs)
        self._debug.outputs['predict_proba'] = y
        return y

    def decision_function(self, X, *args, **kwargs):
        self._debug.inputs['decision_function'] = X
        y = self._debug.methods['decision_function'](self, X, *args, **kwargs)
        self._debug.outputs['decision_function'] = y
        return y

    new_methods = {
        'decision_function': decision_function,
        'transform': transform,
        'predict': predict,
        'predict_proba': predict_proba,
    }

    if hasattr(skl_model, '_debug'):
        raise RuntimeError("The same operator cannot be used twice in "
                           "the same pipeline or this method was called "
                           "a second time.")
    skl_model._debug = BaseEstimatorDebugInformation(skl_model)
    for k in skl_model._debug.methods:
        try:
            setattr(skl_model, k, MethodType(new_methods[k], skl_model))
        except AttributeError:
            warnings.warn("Unable to overwrite method '{}' for class "
                          "{}.".format(k, type(skl_model)))


def collect_intermediate_steps(model, *args, **kwargs):
    """
    Converts a scikit-learn model into ONNX with :func:`convert_sklearn`
    and returns intermediate results for each included operator.

    :param model: model or pipeline to convert
    :param args: arguments for :func:`convert_sklearn`
    :param kwargs: optional arguments for :func:`convert_sklearn`

    The model *model* is modified by the function,
    it should be pickled first to be retrieved unaltered.
    This function is used to check every intermediate model in
    a pipeline.
    """
    if 'intermediate' in kwargs:
        if not kwargs['intermediate']:
            raise ValueError("Parameter intermediate must be true.")
        del kwargs['intermediate']

    model_onnx, topology = convert_sklearn(model, *args, intermediate=True,
                                           **kwargs)

    steps = []
    for operator in topology.topological_operator_iterator():
        if operator.raw_operator is None:
            continue
        _alter_model_for_debugging(operator.raw_operator)
        inputs = [i.full_name for i in operator.inputs]
        outputs = [o.full_name for o in operator.outputs]
        steps.append({
            'model': operator.raw_operator,
            'model_onnx': model_onnx,
            'inputs': inputs,
            'outputs': outputs,
            'short_onnx': select_model_inputs_outputs(
                model_onnx, outputs=outputs)
        })
    return steps
