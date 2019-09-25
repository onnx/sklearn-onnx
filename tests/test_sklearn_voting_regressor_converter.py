"""Tests GLMRegressor converter."""

import unittest
import numpy
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType
)
from test_utils import dump_data_and_model


def _fit_model(model, is_int=False):
    X, y = datasets.make_regression(n_features=4, random_state=0)
    if is_int:
        X = X.astype(numpy.int64)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def model_to_test():
    return VotingRegressor([
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor()),
    ])


class TestVotingRegressorConverter(unittest.TestCase):
    def test_model_voting_regression(self):
        model, X = _fit_model(model_to_test())
        model_onnx = convert_sklearn(
            model, "voting regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnVotingRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            comparable_outputs=[0]
        )

    def test_model_voting_regression_int(self):
        model, X = _fit_model(model_to_test(), is_int=True)
        model_onnx = convert_sklearn(
            model, "voting regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnVotingRegressorInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            comparable_outputs=[0]
        )


if __name__ == "__main__":
    unittest.main()
