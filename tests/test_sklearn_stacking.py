"""Tests VotingRegressor converter."""

import unittest
import numpy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
try:
    from sklearn.ensemble import StackingRegressor, StackingClassifier
except ImportError:
    # New in 0.22
    StackingRegressor = None
    StackingClassifier = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import (
    dump_data_and_model, fit_regression_model,
    fit_classification_model
)


def model_to_test_reg():
    estimators = [
        ('dt', DecisionTreeRegressor()),
        ('las', LinearRegression())]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression())
    return stacking_regressor


def model_to_test_cl():
    estimators = [
        ('dt', DecisionTreeClassifier()),
        ('las', LogisticRegression())]
    stacking_regressor = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression())
    return stacking_regressor


class TestStackingConverter(unittest.TestCase):

    @unittest.skipIf(StackingRegressor is None,
                     reason="new in 0.22")
    def _test_model_stacking_regression(self):
        model, X = fit_regression_model(model_to_test_reg())
        model_onnx = convert_sklearn(
            model, "voting regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnStackingRegressor",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            comparable_outputs=[0]
        )

    @unittest.skipIf(StackingRegressor is None,
                     reason="new in 0.22")
    def test_model_stacking_classifier(self):
        model, X = fit_classification_model(
            model_to_test_cl(), n_classes=2)
        model_onnx = convert_sklearn(
            model, "voting regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnStackingRegressor",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            comparable_outputs=[0]
        )


if __name__ == "__main__":
    unittest.main()
