import unittest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    onnx_built_with_ml,
)
from test_utils import (
    dump_data_and_model,
    dump_multiple_classification,
    fit_classification_model,
)


class TestOneVsRestClassifierConverter(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_02(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            first_class=2,
            suffix="F2",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_string(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            verbose=False,
            label_string=True,
            suffix="String",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_float(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 5)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloat",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_int(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_regression_float(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LinearRegression()), 10)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionFloat-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_regression_int(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LinearRegression()), 10, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionInt-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
