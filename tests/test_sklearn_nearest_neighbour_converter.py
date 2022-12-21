# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's KNeighbours Classifier and Regressor converters.
"""
import sys
import warnings
import unittest
import functools
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import __version__ as ort_version
from pandas import DataFrame
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # older versions of scikit-learn
    from sklearn.utils.testing import ignore_warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    KNeighborsRegressor, RadiusNeighborsRegressor,
    KNeighborsClassifier, RadiusNeighborsClassifier,
    NearestNeighbors)
try:
    from sklearn.impute import KNNImputer
    from sklearn.neighbors import (
        KNeighborsTransformer,
        NeighborhoodComponentsAnalysis)
except ImportError:
    # New in 0.22
    KNNImputer = None
    KNeighborsTransformer = None
    NeighborhoodComponentsAnalysis = None
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        NotImplemented as OrtImpl)
except ImportError:
    OrtImpl = RuntimeError
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from skl2onnx.helpers.onnx_helper import (
    enumerate_model_node_outputs, select_model_inputs_outputs)
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession)


def dont_test_radius():
    return (
        pv.Version(ort_version) <= pv.Version("1.3.0") or
        TARGET_OPSET <= 11)


ort_version = ".".join(ort_version.split('.')[:2])


class TestNearestNeighbourConverter(unittest.TestCase):

    @functools.lru_cache(maxsize=1)
    def _get_iris(self):
        iris = datasets.load_iris()
        X = iris.data[::2, :3]
        y = iris.target[::2]
        return X, y

    def _fit_model_binary_classification(self, model):
        X, y = self._get_iris()
        y[y == 2] = 1
        model.fit(X, y)
        return model, X

    def _fit_model_multiclass_classification(self, model, use_string=False):
        X, y = self._get_iris()
        if use_string:
            y = numpy.array(["cl%d" % _ for _ in y])
        model.fit(X, y)
        return model, X

    @functools.lru_cache(maxsize=20)
    def _get_reg_data(self, n, n_features, n_targets, n_informative=10):
        X, y = datasets.make_regression(
            n, n_features=n_features, random_state=0,
            n_targets=n_targets, n_informative=n_informative)
        return X, y

    def _fit_model(self, model, n_targets=1, label_int=False,
                   n_informative=10):
        X, y = self._get_reg_data(20, 4, n_targets, n_informative)
        if label_int:
            y = y.astype(numpy.int64)
        model.fit(X, y)
        return model, X

    def _fit_model_simple(self, model, n_targets=1, label_int=False,
                          n_informative=3):
        X, y = self._get_reg_data(20, 2, n_targets, n_informative)
        y /= 100
        if label_int:
            y = y.astype(numpy.int64)
        model.fit(X, y)
        return model, X

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor-Dec4")
        dump_data_and_model(
            (X + 0.1).astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor-Dec4")

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.8.0"),
        reason="produces nan values")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_radius(self):
        model, X = self._fit_model(RadiusNeighborsRegressor())
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET,
                                     options={id(model): {'optim': 'cdist'}})
        sess = InferenceSession(
            model_onnx.SerializeToString(),
            providers=["CPUExecutionProvider"])
        X = X[:5]
        got = sess.run(None, {'input': X.astype(numpy.float32)})[0]
        exp = model.predict(X.astype(numpy.float32))
        if any(numpy.isnan(got.ravel())):
            # The model is unexpectedly producing nan values
            # not on all platforms.
            rows = ['--EXP--', str(exp), '--GOT--', str(got),
                    '--EVERY-OUTPUT--']
            for out in enumerate_model_node_outputs(
                    model_onnx, add_node=False):
                onx = select_model_inputs_outputs(model_onnx, out)
                sess = InferenceSession(
                    onx.SerializeToString(),
                    providers=["CPUExecutionProvider"])
                res = sess.run(
                    None, {'input': X.astype(numpy.float32)})
                rows.append('--{}--'.format(out))
                rows.append(str(res))
            if (pv.Version(ort_version) <
                    pv.Version("1.4.0")):
                return
            raise AssertionError('\n'.join(rows))
        assert_almost_equal(exp.ravel(), got.ravel(), decimal=3)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_double(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(
            model, "KNN regressor",
            [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'optim': 'cdist'}})
        self.assertIsNotNone(model_onnx)
        try:
            InferenceSession(
                model_onnx.SerializeToString(),
                providers=["CPUExecutionProvider"])
        except OrtImpl as e:
            if ("Could not find an implementation for the node "
                    "To_TopK:TopK(11)") in str(e):
                # onnxruntime does not declare TopK(11) for double
                return
            raise e
        dump_data_and_model(
            X.astype(numpy.float64)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor64")

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.7.0"),
        reason="nan may happen during computation")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_double_radius(self):
        model, X = self._fit_model(RadiusNeighborsRegressor())
        model_onnx = convert_sklearn(
            model, "KNN regressor",
            [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'optim': 'cdist'}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float64)[:7],
            model, model_onnx,
            basename="SklearnRadiusNeighborsRegressor64")
        dump_data_and_model(
            (X + 10.).astype(numpy.float64)[:7],
            model, model_onnx,
            basename="SklearnRadiusNeighborsRegressor64")

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_yint(self):
        model, X = self._fit_model(
            KNeighborsRegressor(n_neighbors=2), label_int=True)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorYInt")

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_yint_radius(self):
        model, X = self._fit_model(
            RadiusNeighborsRegressor(), label_int=True)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnRadiusNeighborsRegressorYInt")

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_1(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=1),
                                   n_targets=2)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:3],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor2")

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_1_radius(self):
        model, X = self._fit_model_simple(
            RadiusNeighborsRegressor(algorithm="brute"),
            n_targets=2)
        X = X[:-1]
        model_onnx = convert_sklearn(
            model, "KNN regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(
            model_onnx.SerializeToString(),
            providers=["CPUExecutionProvider"])
        got = sess.run(None, {'input': X.astype(numpy.float32)})[0]
        exp = model.predict(X.astype(numpy.float32))
        if any(numpy.isnan(got.ravel())):
            # The model is unexpectedly producing nan values
            # not on all platforms.
            # It happens when two matrices are multiplied,
            # one is (2, 20, 20), second is (20, 20)
            # and contains only 0 or 1 values.
            # The output contains nan values on the first row
            # but not on the second one.
            rows = ['--EXP--', str(exp), '--GOT--', str(got),
                    '--EVERY-OUTPUT--']
            for out in enumerate_model_node_outputs(
                    model_onnx, add_node=False):
                onx = select_model_inputs_outputs(model_onnx, out)
                sess = InferenceSession(
                    onx.SerializeToString(),
                    providers=["CPUExecutionProvider"])
                res = sess.run(
                    None, {'input': X.astype(numpy.float32)})
                rows.append('--{}--'.format(out))
                rows.append(str(res))
            if (ort_version.startswith('1.4.') or
                    ort_version.startswith('1.5.')):
                # TODO: investigate the regression in onnxruntime 1.4
                # One broadcasted multiplication unexpectedly produces nan.
                whole = '\n'.join(rows)
                if "[        nan" in whole:
                    warnings.warn(whole)
                    return
                raise AssertionError(whole)
            if (ort_version.startswith('1.3.') and
                    sys.platform == 'win32'):
                # Same error but different line number for further
                # investigation.
                raise AssertionError(whole)
            raise AssertionError('\n'.join(rows))
        assert_almost_equal(exp, got, decimal=5)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_1_opset(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=1),
                                   n_targets=2)
        for op in [TARGET_OPSET, 12, 11, 10, 9]:
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model, "KNN regressor",
                    [("input", FloatTensorType([None, 4]))],
                    target_opset=op)
                self.assertIsNotNone(model_onnx)
                dump_data_and_model(
                    X.astype(numpy.float32)[:3],
                    model, model_onnx,
                    basename="SklearnKNeighborsRegressor2%d" % op)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_2(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2),
                                   n_targets=2)
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model, model_onnx,
            basename="SklearnKNeighborsRegressor2")

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < 9,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_weights_distance_11(self):
        model, X = self._fit_model(
            KNeighborsRegressor(
                weights="distance", algorithm="brute", n_neighbors=1))
        for op in sorted(set([9, 10, 11, 12, TARGET_OPSET])):
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model, "KNN regressor",
                    [("input", FloatTensorType([None, 4]))],
                    target_opset=op)
                if op < 12 and model_onnx.ir_version > 6:
                    raise AssertionError(
                        "ir_version ({}, op={}) must be <= 6.".format(
                            model_onnx.ir_version, op))
                if op < 11 and model_onnx.ir_version > 5:
                    raise AssertionError(
                        "ir_version ({}, op={}) must be <= 5.".format(
                            model_onnx.ir_version, op))
                if op < 10 and model_onnx.ir_version > 4:
                    raise AssertionError(
                        "ir_version ({}, op={}) must be <= 4.".format(
                            model_onnx.ir_version, op))
                self.assertIsNotNone(model_onnx)
                dump_data_and_model(
                    X.astype(numpy.float32)[:3],
                    model, model_onnx,
                    basename="SklearnKNeighborsRegressorWDist%d-Dec3" % op)

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_weights_distance_11_radius(self):
        model, X = self._fit_model_simple(
            RadiusNeighborsRegressor(
                weights="distance", algorithm="brute", radius=100))
        for op in sorted(set([TARGET_OPSET, 12, 11])):
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model, "KNN regressor",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    target_opset=op)
                self.assertIsNotNone(model_onnx)
                sess = InferenceSession(
                    model_onnx.SerializeToString(),
                    providers=["CPUExecutionProvider"])
                got = sess.run(None, {'input': X.astype(numpy.float32)})[0]
                exp = model.predict(X.astype(numpy.float32))
                assert_almost_equal(exp, got.ravel(), decimal=3)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_metric_cityblock(self):
        model, X = self._fit_model(KNeighborsRegressor(metric="cityblock"))
        model_onnx = convert_sklearn(model, "KNN regressor",
                                     [("input", FloatTensorType([None, 4]))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorMetricCityblock")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < TARGET_OPSET,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
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

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(TARGET_OPSET < 12,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_binary_class_radius(self):
        model, X = self._fit_model_binary_classification(
            RadiusNeighborsClassifier())
        model_onnx = convert_sklearn(
            model, "KNN classifier binary",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model, model_onnx,
            basename="SklearnRadiusNeighborsClassifierBinary")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(TARGET_OPSET < 12,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_multi_class_radius(self):
        model, X = self._fit_model_multiclass_classification(
            RadiusNeighborsClassifier())
        model_onnx = convert_sklearn(
            model, "KNN classifier multi-class",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'optim': 'cdist'}})
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:5],
            model, model_onnx,
            basename="SklearnRadiusNeighborsClassifierMulti")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            KNeighborsClassifier(), n_classes=7, n_labels=3,
            n_samples=100, n_features=10)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KNN Classifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test[:10], model, model_onnx,
            basename="SklearnKNNClassifierMultiLabel-Out0")

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_int(self):
        model, X = self._fit_model(KNeighborsRegressor())
        X = X.astype(numpy.int64)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnKNNRegressorInt-Dec4"
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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
            [('input', Int64TensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET)
        exp = model.predict(X_test)

        sess = InferenceSession(
            model_onnx.SerializeToString(),
            providers=["CPUExecutionProvider"])
        res = sess.run(None, {'input': numpy.array(X_test)})[0].ravel()

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
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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
            basename="SklearnKNNMultiNoCl", verbose=False)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
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
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_test_knn_transform(self):
        iris = datasets.load_iris()
        X, _ = iris.data, iris.target

        X_train, X_test = train_test_split(X, random_state=11)
        clr = NearestNeighbors(n_neighbors=3, radius=None)
        clr.fit(X_train)

        for to in (9, 10, 11):
            if to > TARGET_OPSET:
                break
            model_def = to_onnx(clr, X_train.astype(numpy.float32),
                                target_opset=to)
            oinf = InferenceSession(
                model_def.SerializeToString(),
                providers=["CPUExecutionProvider"])

            X_test = X_test[:3]
            y = oinf.run(None, {'X': X_test.astype(numpy.float32)})
            dist, ind = clr.kneighbors(X_test)

            assert_almost_equal(dist, DataFrame(y[1]).values, decimal=5)
            assert_almost_equal(ind, y[0])

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_default(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(random_state=42), 3)
        model_onnx = convert_sklearn(
            model, "NCA",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnNCADefault")

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_identity(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                init='identity', max_iter=4, random_state=42), 3)
        model_onnx = convert_sklearn(
            model, "NCA",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model,
            model_onnx, basename="SklearnNCAIdentity")

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_double(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                n_components=2, max_iter=4, random_state=42), 3)
        X_test = X_test.astype(numpy.float64)
        model_onnx = convert_sklearn(
            model, "NCA",
            [("input", DoubleTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnNCADouble")

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None,
                     reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_int(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                init='pca', max_iter=4, random_state=42), 3, is_int=True)
        model_onnx = convert_sklearn(
            model, "NCA",
            [("input", Int64TensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnNCAInt")

    @unittest.skipIf(KNeighborsTransformer is None,
                     reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_k_neighbours_transformer_distance(self):
        model, X_test = fit_classification_model(
            KNeighborsTransformer(
                n_neighbors=4, mode='distance'), 2)
        model_onnx = convert_sklearn(
            model, "KNN transformer",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnKNNTransformerDistance")

    @unittest.skipIf(KNeighborsTransformer is None,
                     reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_k_neighbours_transformer_connectivity(self):
        model, X_test = fit_classification_model(
            KNeighborsTransformer(
                n_neighbors=3, mode='connectivity'), 3)
        model_onnx = convert_sklearn(
            model, "KNN transformer",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnKNNTransformerConnectivity")

    @unittest.skipIf(KNNImputer is None,
                     reason="new in 0.22")
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_knn_imputer(self):
        x_train = numpy.array(
            [[1, 2, numpy.nan, 12], [3, numpy.nan, 3, 13],
             [1, 4, numpy.nan, 1], [numpy.nan, 4, 3, 12]], dtype=numpy.float32)
        x_test = numpy.array(
            [[1.3, 2.4, numpy.nan, 1], [-1.3, numpy.nan, 3.1, numpy.nan]],
            dtype=numpy.float32)
        model = KNNImputer(n_neighbors=3, metric='nan_euclidean').fit(x_train)
        for opset in [TARGET_OPSET, 9, 10, 11, 12]:
            if opset > TARGET_OPSET:
                continue
            model_onnx = convert_sklearn(
                model, "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=opset)
            self.assertIsNotNone(model_onnx)
            dump_data_and_model(
                x_test, model, model_onnx,
                basename="SklearnKNNImputer%d" % opset)

    @unittest.skipIf(KNNImputer is None,
                     reason="new in 0.22")
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_knn_imputer_cdist(self):
        x_train = numpy.array(
            [[1, 2, numpy.nan, 12], [3, numpy.nan, 3, 13],
             [1, 4, numpy.nan, 1], [numpy.nan, 4, 3, 12]], dtype=numpy.float32)
        x_test = numpy.array(
            [[1.3, 2.4, numpy.nan, 1], [-1.3, numpy.nan, 3.1, numpy.nan]],
            dtype=numpy.float32)
        model = KNNImputer(n_neighbors=3, metric='nan_euclidean').fit(x_train)

        with self.assertRaises(NameError):
            convert_sklearn(
                model, "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=TARGET_OPSET,
                options={id(model): {'optim2': 'cdist'}})

        for opset in [TARGET_OPSET, 12, 11, 10, 9]:
            if opset > TARGET_OPSET:
                continue
            model_onnx = convert_sklearn(
                model, "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=opset,
                options={id(model): {'optim': 'cdist'}})
            self.assertIsNotNone(model_onnx)
            self.assertIn('op_type: "cdist"', str(model_onnx).lower())
            self.assertNotIn('scan', str(model_onnx).lower())
            dump_data_and_model(
                x_test, model, model_onnx,
                basename="SklearnKNNImputer%dcdist" % opset)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_regressor_multi_reg(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y, y + 10]).T
        model = KNeighborsRegressor(
            algorithm='brute', weights='distance', n_neighbors=7)
        model.fit(X[:13], y[:13])
        onx = to_onnx(model, X[:1],
                      options={id(model): {'optim': 'cdist'}},
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, onx,
            basename="SklearnKNeighborsRegressorMReg")

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_regressor_multi_reg_radius(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y, y + 10]).T
        model = KNeighborsRegressor(
            algorithm='brute', weights='distance')
        model.fit(X[:13], y[:13])
        onx = to_onnx(model, X[:1],
                      options={id(model): {'optim': 'cdist'}},
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model, onx,
            basename="SklearnRadiusNeighborsRegressorMReg")
        dump_data_and_model(
            (X + 0.1).astype(numpy.float32)[:7],
            model, onx,
            basename="SklearnRadiusNeighborsRegressorMReg")

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_classifier_multi_reg2_weight(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.int64)
        y = numpy.vstack([(y + 1) % 2, y % 2]).T
        model = KNeighborsClassifier(
            algorithm='brute', weights='distance', n_neighbors=7)
        model.fit(X[:13], y[:13])
        onx = to_onnx(model, X[:1],
                      options={id(model): {'optim': 'cdist',
                                           'zipmap': False}},
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32)[:11],
            model, onx,
            basename="SklearnKNeighborsClassifierMReg2-Out0")

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_classifier_multi_reg2_weight_radius(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.int64)
        y = numpy.vstack([(y + 1) % 2, y % 2]).T
        model = RadiusNeighborsClassifier(
            algorithm='brute', weights='distance')
        model.fit(X[:13], y[:13])
        onx = to_onnx(model, X[:1],
                      options={id(model): {'optim': 'cdist',
                                           'zipmap': False}},
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32)[:11],
            model, onx,
            basename="SklearnRadiusNeighborsClassifierMReg2-Out0")

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"),
        reason="not available")
    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_classifier_multi_reg3_weight(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.int64)
        y = numpy.vstack([y % 2, y % 2, (y + 1) % 2]).T
        model = KNeighborsClassifier(
            algorithm='brute', weights='distance',
            n_neighbors=7)
        model.fit(X[:13], y[:13])
        onx = to_onnx(model, X[:1],
                      options={id(model): {'optim': 'cdist',
                                           'zipmap': False}},
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32)[:11],
            model, onx,
            basename="SklearnKNeighborsClassifierMReg3-Out0")


if __name__ == "__main__":
    TestNearestNeighbourConverter().test_model_knn_classifier_multilabel()
    unittest.main()
