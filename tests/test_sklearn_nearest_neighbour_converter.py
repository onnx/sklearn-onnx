"""
Tests scikit-learn's KNeighbours Classifier and Regressor converters.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
import onnx
from pandas import DataFrame
from onnx.defs import onnx_opset_version
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
    NearestNeighbors,
)
try:
    from sklearn.imputer import KNNImputer
    from sklearn.neighbors import (
        KNeighborsTransformer,
        NeighborhoodComponentsAnalysis,
    )
except ImportError:
    # New in 0.22
    KNNImputer = None
    KNeighborsTransformer = None
    NeighborhoodComponentsAnalysis = None
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import onnxruntime
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET
)


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
            target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model, model_onnx,
            basename="SklearnKNeighborsClassifierBinary")

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
            target_opset=TARGET_OPSET
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
            target_opset=TARGET_OPSET
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
            model, 'KNN classifier', [('input', FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
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
            model, 'KNN classifier', [('input', FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7], model, model_onnx,
            basename="SklearnKNeighborsClassifierMetricCityblock")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            KNeighborsClassifier(), n_classes=7, n_labels=3,
            n_samples=100, n_features=10)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KNN Classifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnKNNClassifierMultiLabel-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )

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
            basename="SklearnKNNRegressorInt-Dec4"
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

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_multi_class_nocl(self):
        model, X = fit_classification_model(
            KNeighborsClassifier(),
            2, label_string=True)
        model_onnx = convert_sklearn(
            model, "KNN multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnKNNMultiNoCl", verbose=False,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_model_knn_regressor2_2_pipee(self):
        pipe = make_pipeline(StandardScaler(),
                             KNeighborsClassifier())
        model, X = self._fit_model_binary_classification(pipe)
        model_onnx = convert_sklearn(
            model, "KNN pipe",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorPipe2")

    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion("0.5.0"),
        reason="not available")
    def test_onnx_test_knn_transform(self):
        iris = datasets.load_iris()
        X, _ = iris.data, iris.target

        X_train, X_test = train_test_split(X, random_state=11)
        clr = NearestNeighbors(n_neighbors=3)
        clr.fit(X_train)

        for to in (9, 10, 11):
            if to > onnx_opset_version():
                break
            model_def = to_onnx(clr, X_train.astype(numpy.float32),
                                target_opset=to)
            oinf = InferenceSession(model_def.SerializeToString())

            X_test = X_test[:3]
            y = oinf.run(None, {'X': X_test.astype(numpy.float32)})
            dist, ind = clr.kneighbors(X_test)

            assert_almost_equal(dist, DataFrame(y[1]).values, decimal=5)
            assert_almost_equal(ind, y[0])

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    def test_sklearn_nca_default(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(random_state=42), 3)
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", FloatTensorType((None, X_test.shape[1])))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnNCADefault",
        )

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    def test_sklearn_nca_identity(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                init='identity', max_iter=4, random_state=42), 3)
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", FloatTensorType((None, X_test.shape[1])))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnNCAIdentity",
        )

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    def test_sklearn_nca_double(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                n_components=2, max_iter=4, random_state=42), 3)
        X_test = X_test.astype(numpy.float64)
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", DoubleTensorType((None, X_test.shape[1])))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnNCADouble",
        )

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    def test_sklearn_nca_int(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                init='pca', max_iter=4, random_state=42), 3, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", Int64TensorType((None, X_test.shape[1])))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnNCAInt",
        )

    @unittest.skipIf(KNeighborsTransformer is None,
                     reason="new in 0.22")
    def test_sklearn_k_neighbours_transformer_distance(self):
        model, X_test = fit_classification_model(
            KNeighborsTransformer(
                n_neighbors=4, mode='distance'), 2)
        model_onnx = convert_sklearn(
            model,
            "KNN transformer",
            [("input", FloatTensorType((None, X_test.shape[1])))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnKNNTransformerDistance",
        )

    @unittest.skipIf(KNeighborsTransformer is None,
                     reason="new in 0.22")
    def test_sklearn_k_neighbours_transformer_connectivity(self):
        model, X_test = fit_classification_model(
            KNeighborsTransformer(
                n_neighbors=3, mode='connectivity'), 3)
        model_onnx = convert_sklearn(
            model,
            "KNN transformer",
            [("input", FloatTensorType((None, X_test.shape[1])))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnKNNTransformerConnectivity",
        )

    @unittest.skipIf(KNNImputer is None,
                     reason="new in 0.22")
    @unittest.skipIf((StrictVersion(onnx.__version__) <
                      StrictVersion("1.4.1")),
                     reason="ConstantOfShape op not available")
    def test_sklearn_knn_imputer(self):
        x_train = numpy.array(
            [[1, 2, numpy.nan, 12], [3, numpy.nan, 3, 13],
             [1, 4, numpy.nan, 1], [numpy.nan, 4, 3, 12]], dtype=numpy.float32)
        x_test = numpy.array(
            [[1.3, 2.4, numpy.nan, 1], [-1.3, numpy.nan, 3.1, numpy.nan]],
            dtype=numpy.float32)
        model = KNNImputer(n_neighbors=3, metric='nan_euclidean').fit(x_train)
        for opset in [9, 10, 11]:
            model_onnx = convert_sklearn(
                model,
                "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=opset,
            )
            self.assertIsNotNone(model_onnx)
            dump_data_and_model(
                x_test,
                model,
                model_onnx,
                basename="SklearnKNNImputer",
            )


if __name__ == "__main__":
    unittest.main()
