# SPDX-License-Identifier: Apache-2.0

import unittest
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    # onnxruntime <= 0.5
    InvalidArgument = RuntimeError
from sklearn.base import ClassifierMixin
from lightgbm import LGBMClassifier, LGBMRegressor, Dataset, train, Booster
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,  # noqa
    calculate_linear_regressor_output_shapes,
)
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm  # noqa
)
import onnxmltools
from onnxmltools.convert.lightgbm._parse import WrappedBooster  # noqa
from skl2onnx import to_onnx
from skl2onnx._parse import (
    _parse_sklearn_classifier, _parse_sklearn_simple_model)

try:
    from test_utils import dump_single_regression
except ImportError:
    import os
    import sys
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), "..", "tests"))
    from test_utils import dump_single_regression
from test_utils import (
    dump_binary_classification, dump_multiple_classification,
    TARGET_OPSET, TARGET_OPSET_ML)


def calculate_lightgbm_output_shapes(operator):
    op = operator.raw_operator
    if hasattr(op, "_model_dict"):
        objective = op._model_dict['objective']
    elif hasattr(op, 'objective_'):
        objective = op.objective_
    else:
        raise RuntimeError(  # pragma: no cover
            "Unable to find attributes '_model_dict' or 'objective_' in "
            "instance of type %r (list of attributes=%r)." % (
                type(op), dir(op)))
    if objective.startswith('binary') or objective.startswith('multiclass'):
        return calculate_linear_classifier_output_shapes(operator)
    if objective.startswith('regression'):  # pragma: no cover
        return calculate_linear_regressor_output_shapes(operator)
    raise NotImplementedError(  # pragma: no cover
        "Objective '{}' is not implemented yet.".format(objective))


def lightgbm_parser(scope, model, inputs, custom_parsers=None):
    if hasattr(model, "fit"):
        raise TypeError(  # pragma: no cover
            "This converter does not apply on type '{}'."
            "".format(type(model)))

    if len(inputs) == 1:
        wrapped = WrappedBooster(model)
        objective = wrapped.get_objective()
        if objective.startswith('binary'):
            wrapped = WrappedLightGbmBoosterClassifier(wrapped)
            return _parse_sklearn_classifier(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        if objective.startswith('multiclass'):
            wrapped = WrappedLightGbmBoosterClassifier(wrapped)
            return _parse_sklearn_classifier(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        if objective.startswith('regression'):  # pragma: no cover
            return _parse_sklearn_simple_model(
                scope, wrapped, inputs, custom_parsers=custom_parsers)
        raise NotImplementedError(  # pragma: no cover
            "Objective '{}' is not implemented yet.".format(objective))

    # Multiple columns
    this_operator = scope.declare_local_operator('LightGBMConcat')
    this_operator.raw_operator = model
    this_operator.inputs = inputs
    var = scope.declare_local_variable(
        'Xlgbm', inputs[0].type.__class__([None, None]))
    this_operator.outputs.append(var)
    return lightgbm_parser(scope, model, this_operator.outputs,
                           custom_parsers=custom_parsers)


class WrappedLightGbmBoosterClassifier(ClassifierMixin):
    """
    Trick to wrap a LGBMClassifier into a class.
    """

    def __init__(self, wrapped):  # pylint: disable=W0231
        for k in {'boosting_type', '_model_dict', '_model_dict_info',
                  'operator_name', 'classes_', 'booster_', 'n_features_',
                  'objective_', 'boosting_type', 'n_features_in_',
                  'n_features_out_'}:
            if hasattr(wrapped, k):
                setattr(self, k, getattr(wrapped, k))


class TestLightGbmTreeEnsembleModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        update_registered_converter(
            LGBMClassifier, 'LightGbmLGBMClassifier',
            calculate_linear_classifier_output_shapes,
            convert_lightgbm, options={
                'zipmap': [True, False, 'columns'], 'nocl': [True, False]})

        update_registered_converter(
            LGBMRegressor, 'LgbmRegressor',
            calculate_linear_regressor_output_shapes,
            convert_lightgbm)

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version('1.11'),
        reason="converter for lightgbm is too old")
    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        dump_binary_classification(
            model,
            target_opset={'': TARGET_OPSET, 'ai.onnx.ml': TARGET_OPSET_ML})
        dump_multiple_classification(
            model,
            target_opset={'': TARGET_OPSET, 'ai.onnx.ml': TARGET_OPSET_ML})

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version('1.11'),
        reason="converter for lightgbm is too old")
    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        dump_single_regression(
            model,
            target_opset={'': TARGET_OPSET, 'ai.onnx.ml': TARGET_OPSET_ML})

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version('1.11'),
        reason="converter for lightgbm is too old")
    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1)
        dump_single_regression(
            model, suffix="1",
            target_opset={'': TARGET_OPSET, 'ai.onnx.ml': TARGET_OPSET_ML})

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version('1.11'),
        reason="converter for lightgbm is too old")
    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        dump_single_regression(
            model, suffix="2",
            target_opset={'': TARGET_OPSET, 'ai.onnx.ml': TARGET_OPSET_ML})

    @unittest.skipIf(
        pv.Version(onnxmltools.__version__) < pv.Version('1.11'),
        reason="converter for lightgbm is too old")
    def test_lightgbm_booster_multi_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2], [-1, 2], [1, -2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1, 2, 2]
        data = Dataset(X, label=y)
        model = train(
            {'boosting_type': 'gbdt', 'objective': 'multiclass',
             'n_estimators': 3, 'min_child_samples': 1, 'num_class': 3},
            data)

        update_registered_converter(
            WrappedLightGbmBoosterClassifier,
            'WrappedLightGbmBoosterClassifier',
            calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser,
            options={'zipmap': [False, True], 'nocl': [False, True]})
        update_registered_converter(
            WrappedBooster, 'WrappedBooster',
            calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser,
            options={'zipmap': [False, True], 'nocl': [False, True]})
        update_registered_converter(
            Booster, 'LightGbmBooster', calculate_lightgbm_output_shapes,
            convert_lightgbm, parser=lightgbm_parser)

        model_onnx = to_onnx(
            model, initial_types=[('X', FloatTensorType([None, 2]))],
            options={WrappedLightGbmBoosterClassifier: {'zipmap': False}},
            target_opset={'': TARGET_OPSET, 'ai.onnx.ml': TARGET_OPSET_ML})

        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(),
                providers=["CPUExecutionProvider"])
        except InvalidArgument as e:
            raise AssertionError(
                "Cannot load model\n%r" % str(model_onnx)) from e
        expected = model.predict(X)
        res = sess.run(None, {'X': X})
        assert_almost_equal(expected, res[1])


if __name__ == "__main__":
    unittest.main()
