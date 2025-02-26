# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's KNeighbours Classifier and Regressor converters.
"""

import math
import numbers
import sys
import warnings
import unittest
import functools
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
from onnx.reference import ReferenceEvaluator
from onnxruntime import __version__ as ort_version
from pandas import DataFrame

try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # older versions of scikit-learn
    from sklearn.utils.testing import ignore_warnings
from sklearn import datasets, __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    KNeighborsRegressor,
    RadiusNeighborsRegressor,
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    NearestNeighbors,
)

try:
    from sklearn.impute import KNNImputer
    from sklearn.neighbors import KNeighborsTransformer, NeighborhoodComponentsAnalysis
except ImportError:
    # New in 0.22
    KNNImputer = None
    KNeighborsTransformer = None
    NeighborhoodComponentsAnalysis = None
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented as OrtImpl
except ImportError:
    OrtImpl = RuntimeError
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.helpers.onnx_helper import (
    enumerate_model_node_outputs,
    select_model_inputs_outputs,
)
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession,
)


def dont_test_radius():
    return pv.Version(ort_version) <= pv.Version("1.3.0") or TARGET_OPSET <= 11


ort_version = ".".join(ort_version.split(".")[:2])
skl_version = ".".join(sklearn_version.split(".")[:2])


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
            n,
            n_features=n_features,
            random_state=0,
            n_targets=n_targets,
            n_informative=n_informative,
        )
        return X, y

    def _fit_model(self, model, n_targets=1, label_int=False, n_informative=10):
        X, y = self._get_reg_data(20, 4, n_targets, n_informative)
        X /= 100
        if label_int:
            y = y.astype(numpy.int64)
        model.fit(X, y)
        return model, X

    def _fit_model_simple(self, model, n_targets=1, label_int=False, n_informative=3):
        X, y = self._get_reg_data(20, 2, n_targets, n_informative)
        y /= 100
        if label_int:
            y = y.astype(numpy.int64)
        model.fit(X, y)
        return model, X

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor-Dec4",
        )
        dump_data_and_model(
            (X + 0.1).astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor-Dec4",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.15.0"), reason="produces nan values"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_radius(self):
        model, X = self._fit_model(RadiusNeighborsRegressor())
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"optim": "cdist"}},
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        X = X[:5]
        got = sess.run(None, {"input": X.astype(numpy.float32)})[0]
        exp = model.predict(X.astype(numpy.float32))
        if any(numpy.isnan(got.ravel())):
            # The model is unexpectedly producing nan values
            # not on all platforms.
            rows = ["--EXP--", str(exp), "--GOT--", str(got), "--EVERY-OUTPUT--"]
            for out in enumerate_model_node_outputs(model_onnx, add_node=False):
                onx = select_model_inputs_outputs(model_onnx, out)
                sess = InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                res = sess.run(None, {"input": X.astype(numpy.float32)})
                rows.append("--{}--".format(out))
                rows.append(str(res))
            if pv.Version(ort_version) < pv.Version("1.4.0"):
                return
            raise AssertionError("\n".join(rows))
        assert_almost_equal(exp.ravel(), got.ravel(), decimal=3)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < 11, reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_double(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"optim": "cdist"}},
        )
        self.assertIsNotNone(model_onnx)
        try:
            InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except OrtImpl as e:
            if (
                "Could not find an implementation for the node To_TopK:TopK(11)"
            ) in str(e):
                # onnxruntime does not declare TopK(11) for double
                return
            raise e
        dump_data_and_model(
            X.astype(numpy.float64)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor64",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.7.0"),
        reason="nan may happen during computation",
    )
    @ignore_warnings(category=(DeprecationWarning, RuntimeWarning, UserWarning))
    def test_model_knn_regressor_double_radius(self):
        model, X = self._fit_model(RadiusNeighborsRegressor(radius=2.0))
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"optim": "cdist"}},
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float64)[:7],
            model,
            model_onnx,
            basename="SklearnRadiusNeighborsRegressor64",
        )
        dump_data_and_model(
            (X + 10.0).astype(numpy.float64)[:7],
            model,
            model_onnx,
            basename="SklearnRadiusNeighborsRegressor64",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_yint(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2), label_int=True)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressorYInt",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=(DeprecationWarning, RuntimeWarning))
    def test_model_knn_regressor_yint_radius(self):
        model, X = self._fit_model(RadiusNeighborsRegressor(radius=2.0), label_int=True)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnRadiusNeighborsRegressorYInt",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_1(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=1), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:3],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor2",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_1_radius(self):
        model, X = self._fit_model_simple(
            RadiusNeighborsRegressor(algorithm="brute"), n_targets=2
        )
        X = X[:-1]
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": X.astype(numpy.float32)})[0]
        exp = model.predict(X.astype(numpy.float32))
        if any(numpy.isnan(got.ravel())):
            # The model is unexpectedly producing nan values
            # not on all platforms.
            # It happens when two matrices are multiplied,
            # one is (2, 20, 20), second is (20, 20)
            # and contains only 0 or 1 values.
            # The output contains nan values on the first row
            # but not on the second one.
            rows = ["--EXP--", str(exp), "--GOT--", str(got), "--EVERY-OUTPUT--"]
            for out in enumerate_model_node_outputs(model_onnx, add_node=False):
                onx = select_model_inputs_outputs(model_onnx, out)
                sess = InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                res = sess.run(None, {"input": X.astype(numpy.float32)})
                rows.append("--{}--".format(out))
                rows.append(str(res))
            if ort_version.startswith("1.4.") or ort_version.startswith("1.5."):
                # TODO: investigate the regression in onnxruntime 1.4
                # One broadcasted multiplication unexpectedly produces nan.
                whole = "\n".join(rows)
                if "[        nan" in whole:
                    warnings.warn(whole)
                    return
                raise AssertionError(whole)
            if ort_version.startswith("1.3.") and sys.platform == "win32":
                # Same error but different line number for further
                # investigation.
                raise AssertionError(whole)
            raise AssertionError("\n".join(rows))
        assert_almost_equal(exp, got, decimal=5)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_1_opset(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=1), n_targets=2)
        for op in [TARGET_OPSET, 12, 11, 10, 9]:
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model,
                    "KNN regressor",
                    [("input", FloatTensorType([None, 4]))],
                    target_opset=op,
                )
                self.assertIsNotNone(model_onnx)
                dump_data_and_model(
                    X.astype(numpy.float32)[:3],
                    model,
                    model_onnx,
                    basename="SklearnKNeighborsRegressor2%d" % op,
                )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_2(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressor2",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < 9, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_weights_distance_11(self):
        model, X = self._fit_model(
            KNeighborsRegressor(weights="distance", algorithm="brute", n_neighbors=1)
        )
        for op in sorted(set([9, 10, 11, 12, TARGET_OPSET])):
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model,
                    "KNN regressor",
                    [("input", FloatTensorType([None, 4]))],
                    target_opset=op,
                )
                if op < 12 and model_onnx.ir_version > 6:
                    raise AssertionError(
                        "ir_version ({}, op={}) must be <= 6.".format(
                            model_onnx.ir_version, op
                        )
                    )
                if op < 11 and model_onnx.ir_version > 5:
                    raise AssertionError(
                        "ir_version ({}, op={}) must be <= 5.".format(
                            model_onnx.ir_version, op
                        )
                    )
                if op < 10 and model_onnx.ir_version > 4:
                    raise AssertionError(
                        "ir_version ({}, op={}) must be <= 4.".format(
                            model_onnx.ir_version, op
                        )
                    )
                self.assertIsNotNone(model_onnx)
                dump_data_and_model(
                    X.astype(numpy.float32)[:3],
                    model,
                    model_onnx,
                    basename="SklearnKNeighborsRegressorWDist%d-Dec3" % op,
                )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_weights_distance_11_radius(self):
        model, X = self._fit_model_simple(
            RadiusNeighborsRegressor(weights="distance", algorithm="brute", radius=100)
        )
        for op in sorted(set([TARGET_OPSET, 12, 11])):
            if op > TARGET_OPSET:
                continue
            with self.subTest(opset=op):
                model_onnx = convert_sklearn(
                    model,
                    "KNN regressor",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    target_opset=op,
                )
                self.assertIsNotNone(model_onnx)
                sess = InferenceSession(
                    model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, {"input": X.astype(numpy.float32)})[0]
                exp = model.predict(X.astype(numpy.float32))
                assert_almost_equal(exp, got.ravel(), decimal=3)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_metric_cityblock(self):
        model, X = self._fit_model(KNeighborsRegressor(metric="cityblock"))
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", FloatTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressorMetricCityblock",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < TARGET_OPSET, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_binary_class(self):
        model, X = self._fit_model_binary_classification(KNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier binary",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierBinary",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(TARGET_OPSET < 12, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_binary_class_radius(self):
        model, X = self._fit_model_binary_classification(RadiusNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier binary",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnRadiusNeighborsClassifierBinary",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_multi_class(self):
        model, X = self._fit_model_multiclass_classification(KNeighborsClassifier())
        model_onnx = convert_sklearn(
            model,
            "KNN classifier multi-class",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierMulti",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @unittest.skipIf(TARGET_OPSET < 12, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_multi_class_radius(self):
        model, X = self._fit_model_multiclass_classification(
            RadiusNeighborsClassifier()
        )
        model_onnx = convert_sklearn(
            model,
            "KNN classifier multi-class",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"optim": "cdist"}},
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:5],
            model,
            model_onnx,
            basename="SklearnRadiusNeighborsClassifierMulti",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_multi_class_string(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier(), use_string=True
        )
        model_onnx = convert_sklearn(
            model,
            "KNN classifier multi-class",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierMulti",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_weights_distance(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier(weights="distance")
        )
        model_onnx = convert_sklearn(
            model,
            "KNN classifier",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierWeightsDistance",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_metric_cityblock(self):
        model, X = self._fit_model_multiclass_classification(
            KNeighborsClassifier(metric="cityblock")
        )
        model_onnx = convert_sklearn(
            model,
            "KNN classifier",
            [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            model_onnx,
            basename="SklearnKNeighborsClassifierMetricCityblock",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            KNeighborsClassifier(),
            n_classes=7,
            n_labels=3,
            n_samples=100,
            n_features=10,
        )
        options = {id(model): {"zipmap": False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn KNN Classifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        assert "zipmap" not in str(model_onnx).lower()
        dump_data_and_model(
            X_test[:10],
            model,
            model_onnx,
            basename="SklearnKNNClassifierMultiLabel-Out0",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_int(self):
        model, X = self._fit_model(KNeighborsRegressor())
        X = X.astype(numpy.int64)
        model_onnx = convert_sklearn(
            model,
            "KNN regressor",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnKNNRegressorInt-Dec4"
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor_equal(self):
        X, y = datasets.make_regression(n_samples=1000, n_features=100, random_state=42)
        X = X.astype(numpy.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )
        model = KNeighborsRegressor(algorithm="brute", metric="manhattan").fit(
            X_train, y_train
        )
        model_onnx = convert_sklearn(
            model,
            "knn",
            [("input", Int64TensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        exp = model.predict(X_test)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, {"input": numpy.array(X_test)})[0].ravel()

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
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_multi_class_nocl(self):
        model, X = fit_classification_model(
            KNeighborsClassifier(), 2, label_string=True
        )
        model_onnx = convert_sklearn(
            model,
            "KNN multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {"nocl": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert "classlabels_strings" not in sonx
        assert "cl0" not in sonx
        dump_data_and_model(
            X,
            model,
            model_onnx,
            classes=model.classes_,
            basename="SklearnKNNMultiNoCl",
            verbose=False,
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_regressor2_2_pipee(self):
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
        model, X = self._fit_model_binary_classification(pipe)
        model_onnx = convert_sklearn(
            model,
            "KNN pipe",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32)[:2],
            model,
            model_onnx,
            basename="SklearnKNeighborsRegressorPipe2",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
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
            model_def = to_onnx(clr, X_train.astype(numpy.float32), target_opset=to)
            oinf = InferenceSession(
                model_def.SerializeToString(), providers=["CPUExecutionProvider"]
            )

            X_test = X_test[:3]
            y = oinf.run(None, {"X": X_test.astype(numpy.float32)})
            dist, ind = clr.kneighbors(X_test)

            assert_almost_equal(dist, DataFrame(y[1]).values, decimal=5)
            assert_almost_equal(ind, y[0])

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_default(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(random_state=42), 3
        )
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnNCADefault")

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(sklearn_version) < pv.Version("1.1.0"), reason="n-d not supported"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_identity(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(
                init="identity", max_iter=4, random_state=42
            ),
            3,
        )
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnNCAIdentity")

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(sklearn_version) < pv.Version("1.1.0"), reason="n-d not supported"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_double(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(n_components=2, max_iter=4, random_state=42),
            3,
        )
        X_test = X_test.astype(numpy.float64)
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", DoubleTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnNCADouble")

    @unittest.skipIf(NeighborhoodComponentsAnalysis is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(sklearn_version) < pv.Version("1.1.0"), reason="n-d not supported"
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_nca_int(self):
        model, X_test = fit_classification_model(
            NeighborhoodComponentsAnalysis(init="pca", max_iter=4, random_state=42),
            3,
            is_int=True,
        )
        model_onnx = convert_sklearn(
            model,
            "NCA",
            [("input", Int64TensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X_test, model, model_onnx, basename="SklearnNCAInt")

    @unittest.skipIf(KNeighborsTransformer is None, reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_k_neighbours_transformer_distance(self):
        model, X_test = fit_classification_model(
            KNeighborsTransformer(n_neighbors=4, mode="distance"), 2
        )
        model_onnx = convert_sklearn(
            model,
            "KNN transformer",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnKNNTransformerDistance"
        )

    @unittest.skipIf(KNeighborsTransformer is None, reason="new in 0.22")
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_k_neighbours_transformer_connectivity(self):
        model, X_test = fit_classification_model(
            KNeighborsTransformer(n_neighbors=3, mode="connectivity"), 3
        )
        model_onnx = convert_sklearn(
            model,
            "KNN transformer",
            [("input", FloatTensorType((None, X_test.shape[1])))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnKNNTransformerConnectivity"
        )

    def _get_torch_knn_imputer(self):
        import torch

        def _get_weights(dist, weights):
            """Get the weights from an array of distances and a parameter ``weights``.

            Assume weights have already been validated.

            Parameters
            ----------
            dist : ndarray
                The input distances.

            weights : {'uniform', 'distance'}, callable or None
                The kind of weighting used.

            Returns
            -------
            weights_arr : array of the same shape as ``dist``
                If ``weights == 'uniform'``, then returns None.
            """
            if weights in (None, "uniform"):
                return None

            if weights == "distance":
                # if user attempts to classify a point that was zero distance from one
                # or more training points, those training points are weighted as 1.0
                # and the other points as 0.0
                dist = 1.0 / dist
                inf_mask = torch.isinf(dist)
                inf_row = torch.any(inf_mask, axis=1)
                dist[inf_row] = inf_mask[inf_row]
                return dist

            if callable(weights):
                return weights(dist)

        def flatnonzero(x):
            return torch.nonzero(torch.reshape(x, (-1,)), as_tuple=True)[0]

        class NanEuclidean(torch.nn.Module):
            """Implements :func:`sklearn.metrics.nan_euclidean_distances`."""

            def __init__(self, squared=False, copy=True):
                super().__init__()
                self.squared = squared
                self.copy = copy

            def forward(self, X, Y):
                X = X.clone()
                Y = Y.to(X.dtype).clone()

                missing_X = torch.isnan(X)
                missing_Y = torch.isnan(Y)

                # set missing values to zero
                X[missing_X] = 0
                Y[missing_Y] = 0

                # Adjust distances for missing values
                XX = X * X
                YY = Y * Y

                distances = -2 * X @ Y.T
                distances += XX.sum(1, keepdim=True) + YY.sum(1, keepdim=True).T

                distances -= XX @ missing_Y.to(X.dtype).T
                distances -= missing_X.to(X.dtype) @ YY.T

                distances = torch.clip(distances, 0, None)

                present_X = 1 - missing_X.to(X.dtype)
                present_Y = ~missing_Y
                present_count = present_X @ present_Y.to(X.dtype).T
                distances[present_count == 0] = torch.nan
                # avoid divide by zero
                present_count = torch.maximum(
                    torch.tensor([1], dtype=present_count.dtype), present_count
                )
                distances /= present_count
                distances *= X.shape[1]

                if not self.squared:
                    distances = distances.sqrt()

                return distances

        def _get_mask(X, value_to_mask):
            return (
                torch.isnan(X)
                if (  # sklearn.utils._missing.is_scalar_nan(value_to_mask)
                    not isinstance(value_to_mask, numbers.Integral)
                    and isinstance(value_to_mask, numbers.Real)
                    and math.isnan(value_to_mask)
                )
                else (value_to_mask == X)
            )

        class SubWeightMatrix(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.weights = weights

            def forward(self, donors_dist):
                weight_matrix = _get_weights(donors_dist, self.weights)
                if weight_matrix is not None:
                    weight_matrix = weight_matrix.clone()
                    weight_matrix[torch.isnan(weight_matrix)] = 0.0
                else:
                    weight_matrix = torch.ones_like(donors_dist)
                    weight_matrix[torch.isnan(donors_dist)] = 0.0
                return weight_matrix

        class SubDonorsIdx(torch.nn.Module):
            def forward(self, dist_pot_donors, n_neighbors):
                xn = torch.nan_to_num(dist_pot_donors, nan=1.0e10)
                tk = torch.topk(xn, n_neighbors, dim=1, largest=False, sorted=True)
                return tk.indices, tk.values

        class MakeNewWeights(torch.nn.Module):
            def forward(self, donors_mask, donors, weight_matrix):
                return donors_mask.to(donors.dtype) * weight_matrix.to(donors.dtype)

        class CalcImpute(torch.nn.Module):
            """Implements :meth:`sklearn.impute.KNNImputer._calc_impute`."""

            def __init__(self, weights):
                super().__init__()
                self._weights = SubWeightMatrix(weights)
                self._donors_idx = SubDonorsIdx()
                self._make_new_neights = MakeNewWeights()

            def _calc_impute(
                self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col
            ):
                donors_idx, donors_dist = self._donors_idx(dist_pot_donors, n_neighbors)
                weight_matrix = self._weights(donors_dist)
                # Retrieve donor values and calculate kNN average
                donors = fit_X_col.take(donors_idx)
                donors_mask = torch.tensor([1], dtype=donors_idx.dtype) - (
                    mask_fit_X_col.take(donors_idx)
                ).to(donors_idx.dtype)

                new_weights = self._make_new_neights(donors_mask, donors, weight_matrix)

                weights_sum = new_weights.sum(axis=1, keepdim=True)
                div = torch.where(
                    weights_sum == 0,
                    torch.tensor([1], dtype=weights_sum.dtype),
                    weights_sum,
                )
                res = (donors * new_weights).sum(axis=1, keepdim=True) / div
                return res.squeeze(dim=1).to(dist_pot_donors.dtype)

            def forward(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
                return self._calc_impute(
                    dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col
                )

        class ColProcessorAllNan(torch.nn.Module):
            def __init__(self, col: int):
                super().__init__()
                self.col = col

            def forward(
                self,
                X,
                dist_subset,
                mask_fit_X,
                _fit_X,
                receivers_idx,
                all_nan_receivers_idx,
                all_nan_dist_mask,
                dist_chunk,
                dist_idx_map,
                potential_donors_idx,
            ):
                col = self.col
                X = X.clone()
                mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
                mask_sum = mask_.to(X.dtype).sum()

                col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
                div = torch.where(
                    mask_sum > 0, mask_sum, torch.tensor([1], dtype=mask_sum.dtype)
                )
                X[all_nan_receivers_idx, col] = col_sum / div

                # receivers with at least one defined distance
                receivers_idx = receivers_idx[~all_nan_dist_mask]
                dist_subset = dist_chunk[dist_idx_map[receivers_idx]][
                    :, potential_donors_idx
                ]
                return X, dist_subset, receivers_idx

        class ColProcessorIdentity(torch.nn.Module):
            def forward(
                self,
                X,
                dist_subset,
                mask_fit_X,
                _fit_X,
                receivers_idx,
                all_nan_receivers_idx,
                all_nan_dist_mask,
                dist_chunk,
                dist_idx_map,
                potential_donors_idx,
            ):
                return (
                    X,
                    dist_subset,
                    receivers_idx,
                )

        class ColProcessorCond(torch.nn.Module):
            def __init__(self, col: int):
                super().__init__()
                self.col = col
                self._all_nan = ColProcessorAllNan(col)
                self._identity = ColProcessorIdentity()

            def forward(
                self,
                X,
                dist_subset,
                mask_fit_X,
                _fit_X,
                receivers_idx,
                all_nan_receivers_idx,
                all_nan_dist_mask,
                dist_chunk,
                dist_idx_map,
                potential_donors_idx,
            ):
                X, dist_subset, receivers_idx = torch.cond(
                    all_nan_receivers_idx.numel() > 0,
                    self._all_nan,
                    self._identity,
                    [
                        X,
                        dist_subset,
                        mask_fit_X,
                        _fit_X,
                        receivers_idx,
                        all_nan_receivers_idx,
                        all_nan_dist_mask,
                        dist_chunk,
                        dist_idx_map,
                        potential_donors_idx,
                    ],
                )
                return X, dist_subset, receivers_idx

        class ColProcessor(torch.nn.Module):
            """Processes one column (= one feature)."""

            def __init__(self, col, n_neighbors, weights):
                super().__init__()
                self._calc_impute = CalcImpute(weights)
                self._col_cond = ColProcessorCond(col)
                self.col = col
                self.n_neighbors = n_neighbors

            def process_one_col(
                self,
                X,
                dist_chunk,
                non_missing_fix_X,
                mask_fit_X,
                dist_idx_map,
                mask,
                row_missing_idx,
                _fit_X,
            ):
                col = self.col
                X = X.clone()
                row_missing_chunk = row_missing_idx
                col_mask = mask[row_missing_chunk, col]

                potential_donors_idx = torch.nonzero(
                    non_missing_fix_X[:, col], as_tuple=True
                )[0]

                # receivers_idx are indices in X
                # if col_mask is all False, no need to continue
                flat_index = flatnonzero(col_mask)

                receivers_idx = row_missing_chunk[flat_index]

                # distances for samples that needed imputation for column
                dist_subset = dist_chunk[dist_idx_map[receivers_idx]][
                    :, potential_donors_idx
                ]

                # receivers with all nan distances impute with mean
                all_nan_dist_mask = torch.isnan(dist_subset).all(axis=1)
                all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

                # when all_nan_receivers_idx is not empty (training set is small)
                # ... if all_nan_receivers_idx.size > 0:
                #    # onnxruntime does not like this part when it is empty
                #    mask_ = (~mask_fit_X[:, col]).to(_fit_X.dtype)
                #    mask_sum = mask_.to(X.dtype).sum()
                #
                #    col_sum = (_fit_X[mask_ == 1, col]).sum().to(X.dtype)
                #    div = torch.where(mask_sum > 0, mask_sum,
                #                       torch.tensor([1], dtype=mask_sum.dtype))
                #    X[all_nan_receivers_idx, col] = col_sum / div
                #
                #     # receivers with at least one defined distance
                #     receivers_idx = receivers_idx[~all_nan_dist_mask]
                #     dist_subset = dist_chunk
                #           [dist_idx_map[receivers_idx]][:, potential_donors_idx]
                # else
                #     ... identity
                X, dist_subset, receivers_idx = self._col_cond(
                    X,
                    dist_subset,
                    mask_fit_X,
                    _fit_X,
                    receivers_idx,
                    all_nan_receivers_idx,
                    all_nan_dist_mask,
                    dist_chunk,
                    dist_idx_map,
                    potential_donors_idx,
                )

                # when all_nan_receivers_idx is not empty (training set is big)
                tn = torch.tensor(self.n_neighbors)
                n_neighbors = torch.where(
                    tn < potential_donors_idx.shape[0],
                    tn,
                    potential_donors_idx.shape[0],
                )
                # to make sure n_neighbors > 0
                n_neighbors = torch.where(
                    n_neighbors <= 0,
                    torch.tensor([1], dtype=n_neighbors.dtype),
                    n_neighbors,
                )
                value = self._calc_impute(
                    dist_subset,
                    n_neighbors,
                    _fit_X[potential_donors_idx, col],
                    mask_fit_X[potential_donors_idx, col],
                )
                X[receivers_idx, col] = value.to(X.dtype)
                return X

            def forward(
                self,
                X,
                dist_chunk,
                non_missing_fix_X,
                mask_fit_X,
                dist_idx_map,
                mask,
                row_missing_idx,
                _fit_X,
            ):
                return self.process_one_col(
                    X,
                    dist_chunk,
                    non_missing_fix_X,
                    mask_fit_X,
                    dist_idx_map,
                    mask,
                    row_missing_idx,
                    _fit_X,
                )

        class MakeDictIdxMap(torch.nn.Module):
            def forward(self, X, row_missing_idx):
                dist_idx_map = torch.zeros(X.shape[0], dtype=int)
                dist_idx_map[row_missing_idx] = torch.arange(row_missing_idx.shape[0])
                return dist_idx_map

        class TorchKNNImputer(torch.nn.Module):
            def __init__(self, knn_imputer):
                super().__init__()
                assert (
                    knn_imputer.metric == "nan_euclidean"
                ), f"Not implemented for metric={knn_imputer.metric!r}"
                self.dist = NanEuclidean()
                cols = []
                for col in range(knn_imputer._fit_X.shape[1]):
                    cols.append(
                        ColProcessor(col, knn_imputer.n_neighbors, knn_imputer.weights)
                    )
                self.columns = torch.nn.ModuleList(cols)
                # refactoring
                self._make_dict_idx_map = MakeDictIdxMap()
                # knn imputer
                self.missing_values = knn_imputer.missing_values
                self.n_neighbors = knn_imputer.n_neighbors
                self.weights = knn_imputer.weights
                self.metric = knn_imputer.metric
                self.keep_empty_features = knn_imputer.keep_empty_features
                self.add_indicator = knn_imputer.add_indicator
                # results of fitting
                self.indicator_ = knn_imputer.indicator_
                # The training results.
                # self._fit_X = torch.from_numpy(knn_imputer._fit_X.astype(np.float32))
                # self._mask_fit_X = torch.from_numpy(knn_imputer._mask_fit_X)
                # self._valid_mask = torch.from_numpy(knn_imputer._valid_mask)

            def _transform_indicator(self, X):
                if self.add_indicator:
                    if not hasattr(self, "indicator_"):
                        raise ValueError(
                            "Make sure to call _fit_indicator before _transform_indicator"
                        )
                    raise NotImplementedError(type(self.indicator_))
                    # return self.indicator_.transform(X)
                return None

            def _concatenate_indicator(self, X_imputed, X_indicator):
                if not self.add_indicator:
                    return X_imputed
                if X_indicator is None:
                    raise ValueError(
                        "Data from the missing indicator are not provided. Call "
                        "_fit_indicator and _transform_indicator in the imputer "
                        "implementation."
                    )
                return torch.cat([X_imputed, X_indicator], dim=0)

            def transform(self, mask_fit_X, _valid_mask, _fit_X, X):
                X = X.clone()
                mask = _get_mask(X, self.missing_values)

                X_indicator = self._transform_indicator(mask)

                row_missing_idx = flatnonzero(mask[:, _valid_mask].any(axis=1))
                non_missing_fix_X = torch.logical_not(mask_fit_X)

                # Maps from indices from X to indices in dist matrix
                dist_idx_map = self._make_dict_idx_map(X, row_missing_idx)

                # process in fixed-memory chunks
                pairwise_distances = self.dist(X[row_missing_idx, :], _fit_X)

                # The export unfold the loop as it depends on the number of features.
                # Fixed in this case.
                for col_processor in self.columns:
                    X = col_processor(
                        X,
                        pairwise_distances,
                        non_missing_fix_X,
                        mask_fit_X,
                        dist_idx_map,
                        mask,
                        row_missing_idx,
                        _fit_X,
                    )

                if self.keep_empty_features:
                    Xc = X.clone()
                    Xc[:, ~_valid_mask] = 0
                else:
                    Xc = X[:, _valid_mask]

                return self._concatenate_indicator(Xc, X_indicator)

            def forward(self, _mask_fit_X, _valid_mask, _fit_X, X):
                return self.transform(_mask_fit_X, _valid_mask, _fit_X, X)

        return TorchKNNImputer, NanEuclidean

    @unittest.skipIf(KNNImputer is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.16.0"),
        reason="onnxruntime not recent enough",
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_knn_imputer_main(self):
        x_train = numpy.array(
            [
                [1, 2, numpy.nan, 12],
                [3, numpy.nan, 3, 13],
                [1, 4, numpy.nan, 1],
                [numpy.nan, 4, 3, 12],
            ],
            dtype=numpy.float32,
        )
        x_test = numpy.array(
            [[1.3, 2.4, numpy.nan, 1], [-1.3, numpy.nan, 3.1, numpy.nan]],
            dtype=numpy.float32,
        )
        model = KNNImputer(n_neighbors=3, metric="nan_euclidean").fit(x_train)
        for opset in [TARGET_OPSET, 18]:
            if opset > TARGET_OPSET:
                continue
            model_onnx = convert_sklearn(
                model,
                "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=opset,
            )

            try:
                import torch

                has_torch = True
            except ImportError:
                has_torch = False

            if has_torch:
                # DEBUG
                with open("debug.onnx", "wb") as f:
                    f.write(model_onnx.SerializeToString())

                from sklearn.metrics.pairwise import nan_euclidean_distances

                tmodel_cls, dist_cls = self._get_torch_knn_imputer()
                tmodel = tmodel_cls(model)
                mm = dist_cls()
                td = mm(
                    torch.from_numpy(x_test),
                    torch.from_numpy(model._fit_X.astype(numpy.float32)),
                )
                skl = nan_euclidean_distances(
                    x_test, model._fit_X.astype(numpy.float32)
                )
                assert_almost_equal(td.numpy(), skl, decimal=3)

                ty = tmodel.transform(
                    torch.from_numpy(model._mask_fit_X),
                    torch.from_numpy(model._valid_mask),
                    torch.from_numpy(model._fit_X.astype(numpy.float32)),
                    torch.from_numpy(x_test),
                )
                skl = model.transform(x_test)
                assert_almost_equal(ty.numpy(), skl, decimal=3)

                from experimental_experiment.reference import ExtendedReferenceEvaluator

                ExtendedReferenceEvaluator(model_onnx, verbose=0).run(
                    None, {"input": x_test}
                )

                # let's inline first
                import onnx.inliner

                inlined = onnx.inliner.inline_local_functions(model_onnx)
                with open("debug_inlined.onnx", "wb") as f:
                    f.write(inlined.SerializeToString())

                import onnxruntime

                opts = onnxruntime.SessionOptions()
                # opts.log_severity_level = 0
                # opts.log_verbosity_level = 0
                # opts.graph_optimization_level = (
                #     onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                # )
                sess = onnxruntime.InferenceSession(
                    inlined.SerializeToString(),
                    opts,
                    providers=["CPUExecutionProvider"],
                )
                runopts = onnxruntime.RunOptions()
                # runopts.log_severity_level = 0
                # runopts.log_verbosity_level = 0
                sess.run(None, {"input": x_test}, runopts)

            dump_data_and_model(
                x_test,
                model,
                model_onnx,
                basename="SklearnKNNImputer%d" % opset,
                backend="onnxruntime",
                verbose=0,
            )

    @unittest.skipIf(KNNImputer is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.16.0"),
        reason="onnxruntime not recent enough",
    )
    @ignore_warnings(category=DeprecationWarning)
    def test_sklearn_knn_imputer_cdist(self):
        x_train = numpy.array(
            [
                [1, 2, numpy.nan, 12],
                [3, numpy.nan, 3, 13],
                [1, 4, numpy.nan, 1],
                [numpy.nan, 4, 3, 12],
            ],
            dtype=numpy.float32,
        )
        x_test = numpy.array(
            [[1.3, 2.4, numpy.nan, 1], [-1.3, numpy.nan, 3.1, numpy.nan]],
            dtype=numpy.float32,
        )
        model = KNNImputer(n_neighbors=3, metric="nan_euclidean").fit(x_train)

        with self.assertRaises(NameError):
            convert_sklearn(
                model,
                "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=TARGET_OPSET,
                options={id(model): {"optim2": "cdist"}},
            )

        for opset in [TARGET_OPSET, 18]:
            if opset > TARGET_OPSET:
                continue
            model_onnx = convert_sklearn(
                model,
                "KNN imputer",
                [("input", FloatTensorType((None, x_test.shape[1])))],
                target_opset=opset,
                options={id(model): {"optim": "cdist"}},
            )
            self.assertIsNotNone(model_onnx)
            # self.assertIn('op_type: "cdist"', str(model_onnx).lower())
            self.assertNotIn("scan", str(model_onnx).lower())
            dump_data_and_model(
                x_test,
                model,
                model_onnx,
                basename="SklearnKNNImputer%dcdist" % opset,
                backend="onnxruntime",
            )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < 11, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_regressor_multi_reg(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y, y + 10]).T
        model = KNeighborsRegressor(
            algorithm="brute", weights="distance", n_neighbors=7
        )
        model.fit(X[:13], y[:13])
        onx = to_onnx(
            model,
            X[:1],
            options={id(model): {"optim": "cdist"}},
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            onx,
            basename="SklearnKNeighborsRegressorMReg",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_regressor_multi_reg_radius(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y, y + 10]).T
        model = KNeighborsRegressor(algorithm="brute", weights="distance")
        model.fit(X[:13], y[:13])
        onx = to_onnx(
            model,
            X[:1],
            options={id(model): {"optim": "cdist"}},
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float32)[:7],
            model,
            onx,
            basename="SklearnRadiusNeighborsRegressorMReg",
        )
        dump_data_and_model(
            (X + 0.1).astype(numpy.float32)[:7],
            model,
            onx,
            basename="SklearnRadiusNeighborsRegressorMReg",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < 11, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_classifier_multi_reg2_weight(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.int64)
        y = numpy.vstack([(y + 1) % 2, y % 2]).T
        model = KNeighborsClassifier(
            algorithm="brute", weights="distance", n_neighbors=7
        )
        model.fit(X[:13], y[:13])
        onx = to_onnx(
            model,
            X[:1],
            options={id(model): {"optim": "cdist", "zipmap": False}},
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float32)[:11],
            model,
            onx,
            basename="SklearnKNeighborsClassifierMReg2-Out0",
        )

    @unittest.skipIf(dont_test_radius(), reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_classifier_multi_reg2_weight_radius(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.int64)
        y = numpy.vstack([(y + 1) % 2, y % 2]).T
        model = RadiusNeighborsClassifier(algorithm="brute", weights="distance")
        model.fit(X[:13], y[:13])
        onx = to_onnx(
            model,
            X[:1],
            options={id(model): {"optim": "cdist", "zipmap": False}},
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float32)[:11],
            model,
            onx,
            basename="SklearnRadiusNeighborsClassifierMReg2-Out0",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @unittest.skipIf(TARGET_OPSET < 11, reason="needs higher target_opset")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_knn_iris_classifier_multi_reg3_weight(self):
        iris = datasets.load_iris()
        X = iris.data.astype(numpy.float32)
        y = iris.target.astype(numpy.int64)
        y = numpy.vstack([y % 2, y % 2, (y + 1) % 2]).T
        model = KNeighborsClassifier(
            algorithm="brute", weights="distance", n_neighbors=7
        )
        model.fit(X[:13], y[:13])
        onx = to_onnx(
            model,
            X[:1],
            options={id(model): {"optim": "cdist", "zipmap": False}},
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float32)[:11],
            model,
            onx,
            basename="SklearnKNeighborsClassifierMReg3-Out0",
        )

    @unittest.skipIf(KNNImputer is None, reason="new in 0.22")
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.16.0"),
        reason="onnxruntime not recent enough",
    )
    @ignore_warnings(category=DeprecationWarning)
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.20.0"),
        reason="onnxruntime not recent enough",
    )
    def test_sklearn_knn_imputer_issue_2025(self):
        # This test is about having nan or the fact TopK
        # does not handle largest=1 in opset < 11.
        from onnxruntime import InferenceSession

        data = (numpy.arange(14) + 100).reshape((-1, 2)).astype(float)
        data[:, 0] += 1000
        for i in range(5):
            data[i, i % 2] = numpy.nan
        imputer = KNNImputer(n_neighbors=3, metric="nan_euclidean")
        imputer.fit(data)
        initial_type = [("float_input", FloatTensorType([None, data.shape[1]]))]
        onnx_model = convert_sklearn(imputer, initial_types=initial_type)
        input_data = data.astype(numpy.float32)
        expected = imputer.transform(input_data)
        got = ReferenceEvaluator(onnx_model, verbose=0).run(
            None, {"float_input": input_data}
        )[0]
        assert_almost_equal(expected, got)

        # in case onnruntime fails
        # from experimental_experiment.reference import OrtEval

        # got = OrtEval(onnx_model, verbose=10).run(None, {"float_input": input_data})[0]
        # assert_almost_equal(expected, got)

        got = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"float_input": input_data})[0]
        assert_almost_equal(expected, got)
        dump_data_and_model(
            input_data,
            imputer,
            onnx_model,
            basename="SklearnKNNImputer2025",
            backend="onnxruntime",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
