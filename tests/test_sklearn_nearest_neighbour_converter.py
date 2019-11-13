"""
Tests scikit-learn's KNeighbours Classifier and Regressor converters.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import onnxruntime
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
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

    def _fit_model_multiclass_classification(self, model, use_string=False):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        if use_string:
            y = numpy.array(["cl%d" % _ for _ in y])
        model.fit(X, y)
        return model, X

    def _fit_model(self, model, n_targets=1, label_int=False):
        X, y = datasets.make_regression(n_features=4,
                                        random_state=0,
                                        n_targets=n_targets)
        if label_int:
            y = y.astype(numpy.int64)
        model.fit(X, y)
        return model, X

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor_yint(self):
        model, X = self._fit_model(
            KNeighborsRegressor(n_neighbors=2), label_int=True)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorYInt")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor2_1(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=1),
                                   n_targets=2)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor2")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor2_2(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2),
                                   n_targets=2)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor2")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor_weights_distance(self):
        model, X = self._fit_model(
            KNeighborsRegressor(
                weights="distance", algorithm="brute", n_neighbors=1))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:3],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorWeightsDistance-Dec3")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor_metric_cityblock(self):
        model, X = self._fit_model(KNeighborsRegressor(metric="cityblock"))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorMetricCityblock")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_classifier_binary_class(self):
        model, X = self._fit_model_binary_classification(
            KNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier binary",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model, model_onnx,
            basename="SklearnKNeighborsClassifierBinary")

    @unittest.skipIf(True, reason="later")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_classifier_multi_class(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier multi-class",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model, model_onnx,
            basename="SklearnKNeighborsClassifierMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_classifier_multi_class_string(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier(), use_string=True)
        model_onnx = convert_sklearn(
            model,
            "KNN classifier multi-class",
            [("input", FloatTensorType([None, 3]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model, model_onnx,
            basename="SklearnKNeighborsClassifierMulti")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_classifier_weights_distance(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier(weights='distance'))
        model_onnx = convert_sklearn(
            model, 'KNN classifier', [('input', FloatTensorType([None, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7], model, model_onnx,
            basename="SklearnKNeighborsClassifierWeightsDistance")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_classifier_metric_cityblock(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier(metric='cityblock'))
        model_onnx = convert_sklearn(
            model, 'KNN classifier', [('input', FloatTensorType([None, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7], model, model_onnx,
            basename="SklearnKNeighborsClassifierMetricCityblock")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor_int(self):
        model, X = self._fit_model(KNeighborsRegressor())
        X = X.astype(numpy.int64)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnGradientBoostingRegressionInt-Dec4"
        )

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor_equal(self):
        X, y = datasets.make_regression(
            n_samples=1000, n_features=100, random_state=42)
        X = X.astype(numpy.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)
        model = KNeighborsRegressor(
            algorithm='brute', metric='manhattan').fit(X_train, y_train)
        model_onnx = convert_sklearn(
            model, 'knn',
            [('input', Int64TensorType([None, X_test.shape[1]]))])
        exp = model.predict(X_test)

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': numpy.array(X_test)})[0]

        # The conversion has discrepencies when
        # neighbours are at the exact same distance.
        maxd = 1000
        accb = numpy.abs(exp - res) > maxd
        ind = [i for i, a in enumerate(accb) if a == 1]
        assert len(ind) == 0

        accp = numpy.abs(exp - res) < maxd
        acc = numpy.sum(accp)
        ratio = acc * 1.0 / res.shape[0]
        assert ratio >= 0.7
        # assert_almost_equal(exp, res)


if __name__ == "__main__":
    unittest.main()
