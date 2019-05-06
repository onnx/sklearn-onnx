# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import warnings
from types import MethodType
from .. import convert_sklearn
from ..helpers.onnx_helper import select_model_inputs_outputs


def _alter_model_for_debugging(skl_model):
    """
    Overwrite methods transform, predict or predict_proba
    to collect the last inputs and outputs
    seen in these methods.
    """

    class DebugInformation:
        def __init__(self, model):
            self.inputs = {}
            self.outputs = {}
            self.methods = {}
            for name in ['transform', 'predict', 'predict_proba']:
                if hasattr(model, name):
                    self.methods[name] = getattr(model.__class__, name)

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

    new_methods = {
        'transform': transform,
        'predict': predict,
        'predict_proba': predict_proba
    }

    if hasattr(skl_model, '_debug'):
        raise RuntimeError("The same operator cannot be used twice in "
                           "the same pipeline or this method was called "
                           "a second time.")
    skl_model._debug = DebugInformation(skl_model)
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
