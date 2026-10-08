# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's CalibratedClassifierCV converters
"""

import unittest
import packaging.version as pv
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, assert_array_equal
from onnx import checker
from onnxruntime import SessionOptions
from onnxruntime import __version__ as ort_version
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris, make_classification
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    HistGradientBoostingClassifier = None
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)
from test_utils import (
    dump_data_and_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession,
)

ort_version = ort_version.split("+")[0]


def skl12():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("1.2")


def _onnx121() -> bool:
    import onnx
    import packaging.version as pv

    return pv.Version(onnx.__version__) >= pv.Version("1.21.0")


class TestSklearnCalibratedClassifierCVConverters(unittest.TestCase):
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVMNB",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVFloat",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_float_nozipmap(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVMNB",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            options={id(model): {"zipmap": False}},
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVFloatNoZipMap",
        )

    @ignore_warnings(category=FutureWarning)
    def test_model_calibrated_classifier_cv_sigmoid_int(self):
        data = load_digits()
        X, y = data.data, data.target
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVMNB",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.int64),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVInt-Dec4",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_isotonic_float(self):
        data = load_iris()
        X, y = data.data, data.target
        clf = KNeighborsClassifier().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="isotonic").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCVKNN",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        try:
            dump_data_and_model(
                X.astype(np.float32),
                model,
                model_onnx,
                basename="SklearnCalibratedClassifierCVIsotonicFloat",
            )
        except Exception as e:
            raise AssertionError("Issue with model\n{}".format(model_onnx)) from e

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_binary_mnb(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        clf = MultinomialNB().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVBinaryMNB",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_isotonic_binary_knn(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        clf = KNeighborsClassifier().fit(X, y)
        model = CalibratedClassifierCV(clf, cv=2, method="isotonic").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn CalibratedClassifierCV",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )

        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVIsotonicBinaryKNN",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_logistic_regression(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            estimator=LogisticRegression(), method="sigmoid"
        ).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "unused",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierCVBinaryLogReg",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_rf(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            estimator=RandomForestClassifier(n_estimators=2), method="sigmoid"
        ).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "clarf",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierRF",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_gbt(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(n_estimators=2), method="sigmoid"
        ).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "clarf",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierGBT",
        )

    @unittest.skipIf(HistGradientBoostingClassifier is None, reason="not available")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not _onnx121(), reason="onnx runtime not tested previously")
    def test_model_calibrated_classifier_cv_hgbt(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(max_iter=4), method="sigmoid"
        ).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "clarf",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierHGBT",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_tree(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y > 1] = 1
        model = CalibratedClassifierCV(
            estimator=DecisionTreeClassifier(), method="sigmoid"
        ).fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "clarf",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierDT",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_svc(self):
        data = load_iris()
        X, y = data.data, data.target
        model = CalibratedClassifierCV(estimator=SVC(), method="sigmoid").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "unused",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierSVC",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_linearsvc(self):
        data = load_iris()
        X, y = data.data, data.target
        model = CalibratedClassifierCV(estimator=LinearSVC(), method="sigmoid").fit(
            X, y
        )
        model_onnx = convert_sklearn(
            model,
            "unused",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierLinearSVC",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_linearsvc2(self):
        data = load_iris()
        X, y = data.data, data.target
        y[y == 2] = 0
        self.assertEqual(len(set(y)), 2)
        model = CalibratedClassifierCV(estimator=LinearSVC(), method="sigmoid").fit(
            X, y
        )
        model_onnx = convert_sklearn(
            model,
            "unused",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnCalibratedClassifierLinearSVC2",
        )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(not skl12(), reason="base_estimator")
    def test_model_calibrated_classifier_cv_svc2_binary(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X[:90]
        y = y[:90]
        self.assertEqual(len(set(y)), 2)

        for model_sub in [SVC(probability=False), LogisticRegression()]:
            model_sub.fit(X, y)
            with self.subTest(model=model_sub):
                model = CalibratedClassifierCV(
                    estimator=model_sub, cv=2, method="sigmoid"
                ).fit(X, y)
                model_onnx = convert_sklearn(
                    model,
                    "unused",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET,
                    options={id(model): {"zipmap": False}},
                )

                sess = InferenceSession(
                    model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                if sess is not None:
                    try:
                        res = sess.run(None, {"input": X[:5].astype(np.float32)})
                    except RuntimeError as e:
                        raise AssertionError("runtime failed") from e
                    assert_almost_equal(model.predict_proba(X[:5]), res[1])
                    assert_almost_equal(model.predict(X[:5]), res[0])

                name = model_sub.__class__.__name__
                dump_data_and_model(
                    X.astype(np.float32)[:10],
                    model,
                    model_onnx,
                    basename=f"SklearnCalibratedClassifierBinary{name}SVC2",
                )

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_sigmoid_pipeline(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        X, y = make_classification(n_samples=500, n_features=20, random_state=42)
        X = np.abs(X).astype(np.float32)

        clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
        pipe = Pipeline([("id", FunctionTransformer()), ("clf", clf)])

        for ensemble in [True, False]:
            with self.subTest(ensemble=ensemble):
                cal = CalibratedClassifierCV(pipe, method="sigmoid", ensemble=ensemble)
                cal.fit(X, y)

                model_onnx = convert_sklearn(
                    cal,
                    "unused",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET,
                    options={id(cal): {"zipmap": False}},
                )
                sess = InferenceSession(
                    model_onnx.SerializeToString(),
                    providers=["CPUExecutionProvider"],
                )
                res = sess.run(None, {"input": X[:20]})
                assert_almost_equal(cal.predict_proba(X[:20]), res[1], decimal=4)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("0.5.0"), reason="not available"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_calibrated_classifier_cv_isotonic_interpolation(self):
        # The isotonic calibrator must be interpolated linearly between its
        # thresholds, not rounded to the nearest one (issue #1151).
        X, y = make_classification(
            n_samples=2000,
            n_features=10,
            n_informative=10,
            n_redundant=0,
            random_state=30,
        )
        X = X.astype(np.float32)
        X_train, X_test, y_train = X[:1500], X[1500:], y[:1500]

        for method in ["isotonic", "sigmoid"]:
            for opset in {min(10, TARGET_OPSET), TARGET_OPSET}:
                with self.subTest(method=method, opset=opset):
                    clf = RandomForestClassifier(
                        n_estimators=20, max_depth=10, random_state=1234
                    )
                    model = CalibratedClassifierCV(clf, cv=3, method=method).fit(
                        X_train, y_train
                    )
                    model_onnx = convert_sklearn(
                        model,
                        "unused",
                        [("input", FloatTensorType([None, X.shape[1]]))],
                        target_opset=opset,
                        options={id(model): {"zipmap": False}},
                    )
                    sess = InferenceSession(
                        model_onnx.SerializeToString(),
                        providers=["CPUExecutionProvider"],
                    )
                    res = sess.run(None, {"input": X_test})
                    assert_almost_equal(model.predict_proba(X_test), res[1], decimal=4)


@unittest.skipIf(
    pv.Version(".".join(sklearn_version.split(".")[:2])) < pv.Version("1.8"),
    reason="Temperature scaling requires scikit-learn >= 1.8.",
)
class TestSklearnCalibratedClassifierCVTemperatureConverters(unittest.TestCase):
    @staticmethod
    def _data(n_classes, dtype=np.float32):
        X, y = make_classification(
            n_samples=240,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = np.abs(X).astype(dtype)
        return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    def _check_model(self, model, X, opset=TARGET_OPSET, options=None):
        if X.dtype == np.float64:
            input_type = DoubleTensorType
        elif X.dtype == np.int64:
            input_type = Int64TensorType
        else:
            input_type = FloatTensorType
        model_onnx = convert_sklearn(
            model,
            "temperature",
            [("input", input_type([None, X.shape[1]]))],
            target_opset=opset,
            options={id(model): options or {"zipmap": False}},
        )
        checker.check_model(model_onnx)
        session_options = SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = InferenceSession(
            model_onnx.SerializeToString(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        result = session.run(None, {"input": X})
        assert_array_equal(model.predict(X), result[0])
        if options and options.get("zipmap") is True:
            probabilities = np.array(
                [[row[label] for label in model.classes_] for row in result[1]]
            )
        elif options and options.get("zipmap") == "columns":
            probabilities = np.column_stack(result[1:])
        else:
            probabilities = result[1]
        assert_allclose(model.predict_proba(X), probabilities, rtol=2e-4, atol=2e-6)
        assert_allclose(probabilities.sum(axis=1), 1.0, atol=2e-6)
        self.assertTrue(np.isfinite(probabilities).all())
        return result

    def test_temperature_estimators(self):
        for n_classes in [2, 3]:
            X_train, X_test, y_train, _ = self._data(n_classes)
            for ensemble in [True, False]:
                for estimator in [
                    LogisticRegression(max_iter=1000),
                    LinearSVC(random_state=42, max_iter=1000),
                    SVC(probability=False, random_state=42),
                    SVC(probability=True, random_state=42),
                    NuSVC(nu=0.1, probability=True, random_state=42),
                    RandomForestClassifier(n_estimators=5, random_state=42),
                    ExtraTreesClassifier(n_estimators=5, random_state=42),
                    GradientBoostingClassifier(n_estimators=5, random_state=42),
                    HistGradientBoostingClassifier(max_iter=5, random_state=42),
                    DecisionTreeClassifier(max_depth=4, random_state=42),
                    GaussianNB(),
                    MultinomialNB(),
                    KNeighborsClassifier(n_neighbors=3),
                ]:
                    with self.subTest(
                        n_classes=n_classes, ensemble=ensemble, estimator=estimator
                    ):
                        model = CalibratedClassifierCV(
                            estimator, method="temperature", cv=2, ensemble=ensemble
                        ).fit(X_train, y_train)
                        self._check_model(model, X_test)

    def test_temperature_frozen_estimator(self):
        for n_classes in [2, 3]:
            X_train, X_test, y_train, y_test = self._data(n_classes)
            for estimator in [
                LogisticRegression(max_iter=1000),
                RandomForestClassifier(n_estimators=5, random_state=42),
            ]:
                with self.subTest(n_classes=n_classes, estimator=estimator):
                    estimator.fit(X_train, y_train)
                    model = CalibratedClassifierCV(
                        FrozenEstimator(estimator), method="temperature"
                    ).fit(X_test, y_test)
                    # Use new inputs, distinct from training and calibration.
                    X_eval = X_test + np.float32(0.01)
                    assert_array_equal(estimator.predict(X_eval), model.predict(X_eval))
                    self._check_model(model, X_eval)

    def test_temperature_pipeline(self):
        X_train, X_test, y_train, y_test = self._data(3)
        for estimator in [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(n_estimators=5, random_state=42),
            SVC(probability=True, random_state=42),
        ]:
            for frozen in [True, False]:
                with self.subTest(estimator=estimator, frozen=frozen):
                    pipeline = make_pipeline(StandardScaler(), estimator)
                    if frozen:
                        pipeline.fit(X_train, y_train)
                        model = CalibratedClassifierCV(
                            FrozenEstimator(pipeline), method="temperature"
                        ).fit(X_test, y_test)
                    else:
                        model = CalibratedClassifierCV(
                            pipeline, method="temperature", cv=2
                        ).fit(X_train, y_train)
                    self._check_model(model, X_test + np.float32(0.01))

    def test_temperature_types_and_opsets(self):
        for n_classes in [2, 3]:
            for dtype in [np.float32, np.float64, np.int64]:
                X_train, X_test, y_train, _ = self._data(n_classes, dtype=dtype)
                model = CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000), method="temperature", cv=2
                ).fit(X_train, y_train)
                for opset in {9, 12, 13, TARGET_OPSET}:
                    with self.subTest(n_classes=n_classes, dtype=dtype, opset=opset):
                        self._check_model(model, X_test, opset=opset)

    def test_temperature_output_options(self):
        X_train, X_test, y_train, _ = self._data(3)
        y_train = np.array(["five", "eleven", "forty-two"])[y_train]
        model = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000), method="temperature", cv=2
        ).fit(X_train, y_train)
        for zipmap in [True, False, "columns"]:
            with self.subTest(zipmap=zipmap):
                self._check_model(model, X_test, options={"zipmap": zipmap})
        result = self._check_model(
            model, X_test, options={"zipmap": False, "output_class_labels": True}
        )
        assert_array_equal(result[2], model.classes_)

    def test_temperature_non_contiguous_labels(self):
        X_train, X_test, y_train, _ = self._data(3)
        model = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000), method="temperature", cv=2
        ).fit(X_train, y_train)
        # Temperature fitting in sklearn 1.8 expects encoded targets. Relabel
        # the fitted classifiers to test the converter independently of fitting.
        model.classes_ = np.array([5, 11, 42])
        for calibrated in model.calibrated_classifiers_:
            calibrated.classes = model.classes_
            calibrated.estimator.classes_ = model.classes_
        self._check_model(model, X_test)

    def test_temperature_probability_like_decision_scores(self):
        X_train, X_test, y_train, y_test = self._data(3)
        estimator = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        # A multiclass decision function may look like probabilities. sklearn
        # detects this per batch, even though the response is decision_function.
        estimator.coef_[:] = 0.0
        estimator.intercept_[:] = [0.2, 0.3, 0.5]
        model = CalibratedClassifierCV(
            FrozenEstimator(estimator), method="temperature"
        ).fit(X_test, y_test)
        model.calibrated_classifiers_[0].calibrators[0].beta_ = 1.0
        self._check_model(model, X_test)

        # Now only some rows look like probabilities: the entire batch must
        # remain raw scores, rather than converting individual rows to logs.
        estimator.coef_[:, 0] = [1.0, 0.0, 0.0]
        X_eval = np.zeros((2, X_test.shape[1]), dtype=np.float32)
        X_eval[1, 0] = 0.25
        self._check_model(model, X_eval)
        self._check_model(model, X_eval[:1])

    def test_temperature_extreme_scores_and_zero_probabilities(self):
        X_train, X_test, y_train, y_test = self._data(3)
        estimator = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        model = CalibratedClassifierCV(
            FrozenEstimator(estimator), method="temperature"
        ).fit(X_test, y_test)
        model.calibrated_classifiers_[0].calibrators[0].beta_ = 1.0
        self._check_model(model, X_test * np.float32(1000))

        estimator = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        model = CalibratedClassifierCV(
            FrozenEstimator(estimator), method="temperature"
        ).fit(X_test, y_test)
        self.assertTrue((estimator.predict_proba(X_test) == 0).any())
        for beta in [0.1, 1.0, 10.0]:
            with self.subTest(beta=beta):
                model.calibrated_classifiers_[0].calibrators[0].beta_ = beta
                self._check_model(model, X_test)

    def test_temperature_almost_uniform_probabilities(self):
        X_train, X_test, y_train, y_test = self._data(3, dtype=np.float64)
        X_eval = X_test.astype(np.float32)
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                estimator = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                estimator.coef_ = np.zeros_like(estimator.coef_, dtype=dtype)
                estimator.intercept_ = np.array([-0.0001, 0.0001, 0.0], dtype=dtype)
                model = CalibratedClassifierCV(
                    FrozenEstimator(estimator), method="temperature"
                ).fit(X_eval, y_test)
                model.calibrated_classifiers_[0].calibrators[0].beta_ = dtype(4.54e-5)
                # Match sklearn's float32 ties, while retaining the float64
                # winner when only the exported output is rounded to float32.
                label = 0 if dtype == np.float32 else 1
                assert_array_equal(
                    model.predict(X_eval), np.full(X_eval.shape[0], label)
                )
                self._check_model(model, X_eval)

    def test_frozen_existing_calibration_methods(self):
        X_train, X_test, y_train, y_test = self._data(3, dtype=np.float64)
        estimator = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        for method in ["sigmoid", "isotonic"]:
            with self.subTest(method=method):
                model = CalibratedClassifierCV(
                    FrozenEstimator(estimator), method=method
                ).fit(X_test, y_test)
                self._check_model(model, X_test)


if __name__ == "__main__":
    unittest.main(verbosity=2)
