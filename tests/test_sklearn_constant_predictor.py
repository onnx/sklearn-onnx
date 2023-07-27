# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
import numpy as np
from sklearn.multiclass import _ConstantPredictor
from onnxruntime import __version__ as ort_version
from skl2onnx import to_onnx

from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

from test_utils import dump_data_and_model, TARGET_OPSET

ort_version = ".".join(ort_version.split(".")[:2])


class TestConstantPredictorConverter(unittest.TestCase):
    def test_constant_predictor_float(self):
        model = _ConstantPredictor()
        X = np.array([[1, 2]])
        y = np.array([0])
        model.fit(X, y)
        test_x = np.array([[1, 0], [2, 8]])

        model_onnx = to_onnx(
            model,
            "scikit-learn ConstantPredictor",
            initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        self.assertIsNotNone(model_onnx is not None)
        dump_data_and_model(
            test_x.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnConstantPredictorFloat",
        )

    def test_constant_predictor_double(self):
        model = _ConstantPredictor()
        X = np.array([[1, 2]])
        y = np.array([0])
        model.fit(X, y)
        test_x = np.array([[1, 0], [2, 8]])

        model_onnx = to_onnx(
            model,
            "scikit-learn ConstantPredictor",
            initial_types=[("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        self.assertIsNotNone(model_onnx is not None)
        dump_data_and_model(
            test_x.astype(np.float64),
            model,
            model_onnx,
            basename="SklearnConstantPredictorDouble",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
