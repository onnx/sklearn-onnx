"""
Tests scikit-learn's Label Binariser converter.
"""

import unittest
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnLabelBinariser(unittest.TestCase):
    def test_model_label_binariser_default(self):
        X = np.array([1, 2, 6, 4, 2])
        model = LabelBinarizer().fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label binariser",
            [("input", Int64TensorType([None]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnLabelBinariserDefault",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_label_binariser_neg_label(self):
        X = np.array([1, 2, 6, 4, 2])
        model = LabelBinarizer(neg_label=-101).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label binariser",
            [("input", Int64TensorType([None]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnLabelBinariserNegLabel",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_label_binariser_pos_label(self):
        X = np.array([1, 2, 6, 4, 2])
        model = LabelBinarizer(pos_label=123).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label binariser",
            [("input", Int64TensorType([None]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnLabelBinariserPosLabel",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_label_binariser_neg_pos_label(self):
        X = np.array([1, 2, 6, 4, 2])
        model = LabelBinarizer(neg_label=10, pos_label=20).fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label binariser",
            [("input", Int64TensorType([None]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnLabelBinariserNegPosLabel",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_label_binariser_binary_labels(self):
        X = np.array([1, 0, 0, 0, 1])
        model = LabelBinarizer().fit(X)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label binariser",
            [("input", Int64TensorType([None]))],
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnLabelBinariserBinaryLabels",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
