"""
Tests scikit-learn's binarizer converter.
"""

import unittest
from sklearn.ensemble import GradientBoostingRegressor
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import (
    supported_converters, get_latest_tested_opset_version,
    convert_sklearn
)
from test_utils import fit_regression_model


class TestSupportedConverters(unittest.TestCase):
    def test_converters_list(self):
        names = supported_converters(False)
        assert "SklearnBernoulliNB" in names
        assert len(names) > 35

    def test_sklearn_converters(self):
        names = supported_converters(True)
        assert "BernoulliNB" in names
        assert len(names) > 35

    def test_version(self):
        assert get_latest_tested_opset_version() == 11

    def test_ir_version(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="huber"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        assert "ir_version: 6" in str(model_onnx)


if __name__ == "__main__":
    unittest.main()
