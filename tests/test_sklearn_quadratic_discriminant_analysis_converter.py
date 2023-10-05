# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import sklearn
import unittest
import numpy as np
import packaging.version as pv
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from onnxruntime import __version__ as ort_version
from onnx import __version__ as onnx_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

from test_utils import dump_data_and_model, TARGET_OPSET

ort_version = ".".join(ort_version.split(".")[:2])
onnx_version = ".".join(onnx_version.split(".")[:2])


class TestQuadraticDiscriminantAnalysisConverter(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(sklearn.__version__) < pv.Version("1.0"), reason="scikit-learn<1.0"
    )
    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="fails with onnx 1.10"
    )
    def test_model_qda_2c2f_float(self):
        # 2 classes, 2 features
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [0.8, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(
            X_test.astype(np.float32),
            skl_model,
            onnx_model,
            basename="SklearnQDA_2c2f_Float",
        )

    @unittest.skipIf(
        pv.Version(sklearn.__version__) < pv.Version("1.0"), reason="scikit-learn<1.0"
    )
    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="fails with onnx 1.10"
    )
    def test_model_qda_2c3f_float(self):
        # 2 classes, 3 features
        X = np.array(
            [[-1, -1, 0], [-2, -1, 1], [-3, -2, 0], [1, 1, 0], [2, 1, 1], [3, 2, 1]]
        )
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1, 0], [-1, -1.6, 0], [1, 1.5, 1], [3.1, 2.1, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(
            X_test.astype(np.float32),
            skl_model,
            onnx_model,
            basename="SklearnQDA_2c3f_Float",
        )

    @unittest.skipIf(
        pv.Version(sklearn.__version__) < pv.Version("1.0"), reason="scikit-learn<1.0"
    )
    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="fails with onnx 1.10"
    )
    def test_model_qda_3c2f_float(self):
        # 3 classes, 2 features
        X = np.array(
            [
                [-1, -1],
                [-2, -1],
                [-3, -2],
                [1, 1],
                [2, 1],
                [3, 2],
                [-1, 2],
                [-2, 3],
                [-2, 2],
            ]
        )
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        X_test = np.array([[-0.8, -1], [0.8, 1], [-0.8, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(
            X_test.astype(np.float32),
            skl_model,
            onnx_model,
            basename="SklearnQDA_3c2f_Float",
        )

    @unittest.skipIf(
        pv.Version(sklearn.__version__) < pv.Version("1.0"), reason="scikit-learn<1.0"
    )
    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="fails with onnx 1.10"
    )
    def test_model_qda_2c2f_double(self):
        # 2 classes, 2 features
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]).astype(
            np.double
        )
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1], [0.8, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(
            X_test.astype(np.double),
            skl_model,
            onnx_model,
            basename="SklearnQDA_2c2f_Double",
        )

    @unittest.skipIf(
        pv.Version(sklearn.__version__) < pv.Version("1.0"), reason="scikit-learn<1.0"
    )
    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="fails with onnx 1.10"
    )
    def test_model_qda_2c3f_double(self):
        # 2 classes, 3 features
        X = np.array(
            [[-1, -1, 0], [-2, -1, 1], [-3, -2, 0], [1, 1, 0], [2, 1, 1], [3, 2, 1]]
        ).astype(np.double)
        y = np.array([1, 1, 1, 2, 2, 2])
        X_test = np.array([[-0.8, -1, 0], [-1, -1.6, 0], [1, 1.5, 1], [3.1, 2.1, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(
            X_test.astype(np.double),
            skl_model,
            onnx_model,
            basename="SklearnQDA_2c3f_Double",
        )

    @unittest.skipIf(
        pv.Version(sklearn.__version__) < pv.Version("1.0"), reason="scikit-learn<1.0"
    )
    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="fails with onnx 1.10"
    )
    def test_model_qda_3c2f_double(self):
        # 3 classes, 2 features
        X = np.array(
            [
                [-1, -1],
                [-2, -1],
                [-3, -2],
                [1, 1],
                [2, 1],
                [3, 2],
                [-1, 2],
                [-2, 3],
                [-2, 2],
            ]
        ).astype(np.double)
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        X_test = np.array([[-0.8, -1], [0.8, 1], [-0.8, 1]])

        skl_model = QuadraticDiscriminantAnalysis()
        skl_model.fit(X, y)

        onnx_model = convert_sklearn(
            skl_model,
            "scikit-learn QDA",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        self.assertIsNotNone(onnx_model)
        dump_data_and_model(
            X_test.astype(np.double),
            skl_model,
            onnx_model,
            basename="SklearnQDA_3c2f_Double",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
