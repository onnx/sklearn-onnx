# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's CalibratedClassifierCV converters
"""

import unittest
import packaging.version as pv
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import __version__ as ort_version
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    HistGradientBoostingClassifier = None
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
try:
    from skl2onnx.common._apply_operation import apply_less
except ImportError:
    # onnxconverter-common is too old
    apply_less = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
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
    @unittest.skipIf(not skl12(), reason="base_estimator")
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
    @unittest.skipIf(apply_less is None, reason="onnxconverter-common old")
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
    @unittest.skipIf(apply_less is None, reason="onnxconverter-common old")
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
    @unittest.skipIf(apply_less is None, reason="onnxconverter-common old")
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
    @unittest.skipIf(apply_less is None, reason="onnxconverter-common old")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
