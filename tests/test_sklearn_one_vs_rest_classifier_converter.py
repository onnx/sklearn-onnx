import unittest
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
    TARGET_OPSET
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
            target_opset=TARGET_OPSET
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
            target_opset=TARGET_OPSET
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
            target_opset=TARGET_OPSET
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_float(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression(solver='liblinear')), 3)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
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
    def test_ovr_classification_decision_function(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 4)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationDecisionFunction",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function'],
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_decision_function_binary(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2)
        options = {id(model): {'raw_scores': True}}
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationDecisionFunctionBinary",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
            methods=['predict', 'decision_function_binary'],
        )
        options = {id(model): {'raw_scores': True, 'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'input': X})[1]
        dec = model.decision_function(X)
        assert_almost_equal(got[:, 1], dec, decimal=4)
        assert_almost_equal(-got[:, 0], dec, decimal=4)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_int(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
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
    def test_ovr_classification_float_binary(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloatBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_int_binary(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(LogisticRegression()), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationIntBin",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_float_mlp(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(MLPClassifier()), 5)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloatMLP",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_int_ensemble(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(GradientBoostingClassifier()), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationIntEnsemble",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_float_binary_ensemble(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(GradientBoostingClassifier()), 2)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationFloatBinEnsemble",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_classification_int_binary_mlp(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(MLPClassifier()), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr classification",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRClassificationIntBinMLP",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_regression_float(self):
        """The test is unstable, some observations
        are equidistant to more than one class,
        the chosen is difficult to predict. So we
        check only probabilities."""
        rs = 11
        model, X = fit_classification_model(
            OneVsRestClassifier(
                LinearRegression()), 3, random_state=rs)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X[:5],
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
            target_opset=TARGET_OPSET
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

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_regression_float_mlp(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(MLPRegressor()), 7)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionFloatMLP-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_regression_int_ensemble(self):
        model, X = fit_classification_model(
            OneVsRestClassifier(GradientBoostingRegressor()), 4, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ovr regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnOVRRegressionIntEnsemble-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
