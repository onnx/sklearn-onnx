# SPDX-License-Identifier: Apache-2.0


import unittest
import numbers
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.datasets import load_iris
from skl2onnx import update_registered_converter, convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,  # noqa
    calculate_linear_regressor_output_shapes,
)
from skl2onnx._parse import _parse_sklearn_classifier
from xgboost import XGBRegressor, XGBClassifier
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost  # noqa
)

try:
    from test_utils import dump_single_regression
except ImportError:
    import os
    import sys
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), "..", "tests"))
    from test_utils import dump_single_regression
from test_utils import dump_multiple_classification, TARGET_OPSET


class TestXGBoostModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        def custom_parser(scope, model, inputs, custom_parsers=None):
            if custom_parsers is not None and model in custom_parsers:
                return custom_parsers[model](
                    scope, model, inputs, custom_parsers=custom_parsers)
            if not all(isinstance(i, (numbers.Real, bool, np.bool_))
                       for i in model.classes_):
                raise NotImplementedError(
                    "Current converter does not support string labels.")
            return _parse_sklearn_classifier(scope, model, inputs)

        update_registered_converter(
            XGBClassifier, 'XGBClassifier',
            calculate_linear_classifier_output_shapes,
            convert_xgboost, parser=custom_parser,
            options={'zipmap': [True, False, 'columns'],
                     'nocl': [True, False]})
        update_registered_converter(
            XGBRegressor, 'XGBRegressor',
            calculate_linear_regressor_output_shapes,
            convert_xgboost)

    def test_xgb_regressor(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBRegressor()
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(conv_model is not None)
        dump_single_regression(xgb, suffix="-Dec4")

    def test_xgb_classifier(self):
        xgb = XGBClassifier(n_estimators=2, max_depth=2)
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))],
            options={id(xgb): {'zipmap': False}},
            target_opset=TARGET_OPSET)
        sess = InferenceSession(conv_model.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        assert_almost_equal(xgb.predict_proba(X), res[1])
        assert_almost_equal(xgb.predict(X), res[0])

    def test_xgb_classifier_multi(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb,
            initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(
            xgb, allow_failure="StrictVersion(onnx.__version__) "
            "< StrictVersion('1.3.0')")

    def test_xgb_classifier_multi_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier(objective='reg:logistic')
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(
            xgb, suffix="RegLog",
            allow_failure="StrictVersion(onnx.__version__) < "
            "StrictVersion('1.3.0')")

    def test_xgb_classifier_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0
        xgb = XGBClassifier(objective='binary:logistic')
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))],
            options={id(xgb): {'zipmap': False}},
            target_opset=TARGET_OPSET)
        self.assertTrue(conv_model is not None)
        sess = InferenceSession(conv_model.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        assert_almost_equal(xgb.predict_proba(X), res[1])
        assert_almost_equal(xgb.predict(X), res[0])


if __name__ == "__main__":
    unittest.main()
