# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy
from sklearn.base import ClassifierMixin
from skl2onnx._parse import (
    _parse_sklearn_classifier, _parse_sklearn_simple_model
)
from skl2onnx.common._apply_operation import apply_concat


class WrappedLightGbmBooster:
    """
    A booster can be a classifier, a regressor.
    Trick to wrap it in a minimal function.
    """

    def __init__(self, booster):
        self.booster_ = booster
        self._model_dict = self.booster_.dump_model()
        self.classes_ = self._generate_classes(self._model_dict)
        self.n_features_ = len(self._model_dict['feature_names'])
        if self._model_dict['objective'].startswith('binary'):
            self.operator_name = 'LgbmClassifier'
        elif self._model_dict['objective'].startswith('regression'):
            self.operator_name = 'LgbmRegressor'
        else:
            raise NotImplementedError(
                'Unsupported LightGbm objective: {}'.format(
                    self._model_dict['objective']))
        if self._model_dict.get('average_output', False):
            self.boosting_type = 'rf'
        else:
            # Other than random forest, other boosting types
            # do not affect later conversion.
            # Here `gbdt` is chosen for no reason.
            self.boosting_type = 'gbdt'

    def _generate_classes(self, model_dict):
        if model_dict['num_class'] == 1:
            return numpy.asarray([0, 1])
        return numpy.arange(model_dict['num_class'])


class WrappedLightGbmBoosterClassifier(ClassifierMixin):
    """
    Trick to wrap a LGBMClassifier into a class.
    """

    def __init__(self, wrapped):
        for k in {'boosting_type', '_model_dict', 'operator_name',
                  'classes_', 'booster_', 'n_features_'}:
            setattr(self, k, getattr(wrapped, k))


class MockWrappedLightGbmBoosterClassifier(WrappedLightGbmBoosterClassifier):
    """
    Mocked lightgbm.
    """

    def __init__(self, tree):
        self.dumped_ = tree

    def dump_model(self):
        "mock dump_model method"
        self.visited = True
        return self.dumped_


def lightgbm_parser(scope, model, inputs, custom_parsers=None):
    """
    Agnostic parser for LightGBM Booster.
    """
    if hasattr(model, "fit"):
        raise TypeError("This converter does not apply on type '{}'."
                        "".format(type(model)))

    if len(inputs) == 1:
        wrapped = WrappedLightGbmBooster(model)
        if wrapped._model_dict['objective'].startswith('binary'):
            wrapped = WrappedLightGbmBoosterClassifier(wrapped)
            return _parse_sklearn_classifier(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        if wrapped._model_dict['objective'].startswith('regression'):
            return _parse_sklearn_simple_model(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        raise NotImplementedError(
            "Objective '{}' is not implemented yet.".format(
                wrapped._model_dict['objective']))

    # Multiple columns
    this_operator = scope.declare_local_operator('LightGBMConcat')
    this_operator.inputs = inputs
    var = scope.declare_local_variable(
        'Xlgbm', inputs[0].type.__class__([None, None]))
    this_operator.outputs.append(var)
    return lightgbm_parser(
        scope, model, this_operator.outputs, custom_parsers=custom_parsers)


def shape_calculator_lightgbm_concat(operator):
    """
    Shape calculator for operator *LightGBMConcat*.
    """
    pass


def converter_lightgbm_concat(scope, operator, container):
    """
    Converter for operator *LightGBMConcat*.
    """
    apply_concat(scope, [_.full_name for _ in operator.inputs],
                 operator.outputs[0].full_name,
                 container, axis=1)
