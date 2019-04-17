"""
Tests GLMRegressor converter.
"""
import unittest
import numpy
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


class TestNearestNeighbourConverter(unittest.TestCase):
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

    def _fit_model(self, model, n_targets=1):
        X, y = datasets.make_regression(n_features=4,
                                        random_state=0,
                                        n_targets=n_targets)
        model.fit(X, y)
        return model, X

    def test_model_knn_regressor(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

    def test_model_knn_regressor2_1(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=1),
                                   n_targets=2)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor2-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

    def test_model_knn_regressor2_2(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2),
                                   n_targets=2)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor2-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__) "
            "<= StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

    def test_model_knn_regressor_weights_distance(self):
        model, X = self._fit_model(KNeighborsRegressor(weights="distance"))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressorWeightsDistance-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
            "StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

    def test_model_knn_regressor_metric_cityblock(self):
        model, X = self._fit_model(KNeighborsRegressor(metric="cityblock"))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressorMetricCityblock-OneOffArray",
            allow_failure="StrictVersion(onnxruntime.__version__) <= "
            "StrictVersion('0.2.1') or "
            "StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_knn_classifier_binary_class(self):
        model, X = self._fit_model_binary_classification(
            KNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier binary",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            numpy.atleast_2d(X[0]).astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierBinary",
            allow_failure="StrictVersion(onnx.__version__) "
            "== StrictVersion('1.1.2') or "
            "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1') "
            "or StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_knn_classifier_multi_class(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier multi-class",
            [("input", FloatTensorType([1, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            numpy.atleast_2d(X[0]).astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierMulti",
            allow_failure="StrictVersion(onnx.__version__) "
            "== StrictVersion('1.1.2') or "
            "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1') "
            "or StrictVersion(onnx.__version__) == StrictVersion('1.4.1')",
        )


if __name__ == "__main__":
    unittest.main()
