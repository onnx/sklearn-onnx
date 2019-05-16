import unittest
import numpy
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import LinearSVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


class TestGLMClassifierConverter(unittest.TestCase):
    def _fit_model_binary_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X

    def _fit_model_multiclass_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model, X

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class(self):
        model, X = self._fit_model_binary_classification(
            linear_model.LogisticRegression())
        model_onnx = convert_sklearn(model, "logistic regression",
                                     [("input", FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinary",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_binary_class_nointercept(self):
        model, X = self._fit_model_binary_classification(
            linear_model.LogisticRegression(fit_intercept=False))
        model_onnx = convert_sklearn(model, "logistic regression",
                                     [("input", FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionBinaryNoIntercept",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class(self):
        model, X = self._fit_model_multiclass_classification(
            linear_model.LogisticRegression())
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMulti",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_no_intercept(self):
        model, X = self._fit_model_multiclass_classification(
            linear_model.LogisticRegression(fit_intercept=False))
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiNoIntercept",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_lbfgs(self):
        model, X = self._fit_model_multiclass_classification(
            linear_model.LogisticRegression(solver='lbfgs', penalty='none'))
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiLbfgs",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_liblinear_l1(self):
        model, X = self._fit_model_multiclass_classification(
            linear_model.LogisticRegression(solver='liblinear', penalty='l1'))
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiLiblinearL1",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_logistic_regression_multi_class_saga_elasticnet(self):
        model, X = self._fit_model_multiclass_classification(
            linear_model.LogisticRegression(
                solver='saga', penalty='elasticnet', l1_ratio=0.1))
        model_onnx = convert_sklearn(
            model,
            "multi-class logistic regression",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLogitisticRegressionMultiSagaElasticnet",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_binary_class(self):
        model, X = self._fit_model_binary_classification(LinearSVC())
        model_onnx = convert_sklearn(model, "linear SVC",
                                     [("input", FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLinearSVCBinary-NoProb",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_linear_svc_multi_class(self):
        model, X = self._fit_model_multiclass_classification(LinearSVC())
        model_onnx = convert_sklearn(
            model,
            "multi-class linear SVC",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLinearSVCMulti",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_binary_class(self):
        model, X = self._fit_model_binary_classification(
            linear_model.SGDClassifier())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD binary classifier",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierBinary-NoProb-Dec4",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_sgd_multi_class(self):
        model, X = self._fit_model_multiclass_classification(
            linear_model.SGDClassifier())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD multi-class classifier",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnSGDClassifierMulti-Dec3",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
