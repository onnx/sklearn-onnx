# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's binarizer converter.
"""

import unittest
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import (
    supported_converters, convert_sklearn, to_onnx,
    update_registered_converter)
from skl2onnx.operator_converters.linear_classifier import (
    convert_sklearn_linear_classifier)
from skl2onnx.shape_calculators.linear_classifier import (
    calculate_linear_classifier_output_shapes)
from test_utils import fit_regression_model, TARGET_OPSET


class DummyClassifier(LogisticRegression):
    pass


class TestSupportedConverters(unittest.TestCase):
    def test_converters_list(self):
        names = supported_converters(False)
        assert "SklearnBernoulliNB" in names
        assert len(names) > 35

    def test_sklearn_converters(self):
        names = supported_converters(True)
        assert "BernoulliNB" in names
        assert len(names) > 35

    def test_ir_version(self):
        model, X = fit_regression_model(
            GradientBoostingRegressor(n_estimators=3, loss="huber"))
        model_onnx = convert_sklearn(
            model,
            "gradient boosting regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sub = "ir_version: "
        if sub not in str(model_onnx):
            raise AssertionError(
                "Unable to find '{}' (opset={}) in\n{}".format(
                    sub, TARGET_OPSET, str(model_onnx)))

    def test_register_classifier(self):
        update_registered_converter(
            DummyClassifier, 'DummyClassifierAlias',
            calculate_linear_classifier_output_shapes,
            convert_sklearn_linear_classifier,
            options={'nocl': [True, False],
                     'zipmap': [True, False, 'columns'],
                     'output_class_labels': [False, True],
                     'raw_scores': [True, False]})
        pipe = Pipeline([('st', StandardScaler()), ('d', DummyClassifier())])
        X = np.array([[0, 1], [1, 0], [0.5, 0.5]], dtype=np.float64)
        y = np.array([1, 0, 1], dtype=np.int64)
        pipe.fit(X, y)

        model_onnx = to_onnx(pipe, X.astype(np.float32))
        assert "zipmap" in str(model_onnx).lower()
        model_onnx = to_onnx(pipe, X.astype(np.float32),
                             options={'d__zipmap': False})
        assert "zipmap" not in str(model_onnx).lower()

        model_onnx = to_onnx(
            pipe, X.astype(np.float32),
            options={DummyClassifier: {'zipmap': False,
                                       'output_class_labels': True}})
        assert "zipmap" not in str(model_onnx).lower()
        self.assertEqual(3, len(model_onnx.graph.output))

        model_onnx = to_onnx(
            pipe, X.astype(np.float32),
            options={id(pipe.steps[-1][-1]): {
                'zipmap': False, 'output_class_labels': True}})
        assert "zipmap" not in str(model_onnx).lower()
        self.assertEqual(3, len(model_onnx.graph.output))

        model_onnx = to_onnx(
            pipe, X.astype(np.float32),
            options={'d__zipmap': False, 'd__output_class_labels': True})
        assert "zipmap" not in str(model_onnx).lower()
        self.assertEqual(3, len(model_onnx.graph.output))

        model_onnx = to_onnx(
            pipe, X.astype(np.float32),
            options={'zipmap': False, 'output_class_labels': True})
        assert "zipmap" not in str(model_onnx).lower()
        self.assertEqual(3, len(model_onnx.graph.output))


if __name__ == "__main__":
    unittest.main()
