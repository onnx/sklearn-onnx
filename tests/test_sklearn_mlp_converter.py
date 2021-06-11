# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's MLPClassifier and MLPRegressor converters.
"""
from distutils.version import StrictVersion
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    try:
        from sklearn.utils.testing import ignore_warnings
    except ImportError:
        def ignore_warnings(category=Warning):
            return lambda x: x

from sklearn.exceptions import ConvergenceWarning
from onnxruntime import InferenceSession, __version__ as ort_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model,
    fit_classification_model,
    fit_multilabel_classification_model,
    fit_regression_model,
    TARGET_OPSET
)


class TestSklearnMLPConverters(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_binary(self):
        model, X_test = fit_classification_model(
            MLPClassifier(random_state=42), 2)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierBinary",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multiclass_default(self):
        model, X_test = fit_classification_model(
            MLPClassifier(random_state=42), 4)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClass",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multilabel_default(self):
        model, X_test = fit_multilabel_classification_model(
            MLPClassifier(random_state=42))
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiLabel",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_regressor_default(self):
        model, X_test = fit_regression_model(
            MLPRegressor(random_state=42))
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multiclass_identity(self):
        model, X_test = fit_classification_model(
            MLPClassifier(random_state=42, activation="identity"), 3,
            is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", Int64TensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClassIdentityActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multilabel_identity(self):
        model, X_test = fit_multilabel_classification_model(
            MLPClassifier(random_state=42, activation="identity"),
            is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", Int64TensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiLabelIdentityActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_regressor_identity(self):
        model, X_test = fit_regression_model(
            MLPRegressor(random_state=42, activation="identity"), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", Int64TensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPRegressorIdentityActivation-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multiclass_logistic(self):
        model, X_test = fit_classification_model(
            MLPClassifier(random_state=42, activation="logistic"), 5)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClassLogisticActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multilabel_logistic(self):
        model, X_test = fit_multilabel_classification_model(
            MLPClassifier(random_state=42, activation="logistic"), n_classes=4)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiLabelLogisticActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_regressor_logistic(self):
        model, X_test = fit_regression_model(
            MLPRegressor(random_state=42, activation="logistic"))
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPRegressorLogisticActivation-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multiclass_tanh(self):
        model, X_test = fit_classification_model(
            MLPClassifier(random_state=42, activation="tanh"), 3)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiClassTanhActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_multilabel_tanh(self):
        model, X_test = fit_multilabel_classification_model(
            MLPClassifier(random_state=42, activation="tanh"), n_labels=3)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPClassifierMultiLabelTanhActivation",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_regressor_tanh(self):
        model, X_test = fit_regression_model(
            MLPRegressor(random_state=42, activation="tanh"))
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPRegressorTanhActivation-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_regressor_bool(self):
        model, X_test = fit_regression_model(
            MLPRegressor(random_state=42), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn MLPRegressor",
            [("input", BooleanTensorType([None, X_test.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnMLPRegressorBool",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion('1.0.0'),
        reason="onnxruntime %s" % '1.0.0')
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_model_mlp_classifier_nozipmap(self):
        X, y = make_multilabel_classification(n_labels=5, n_classes=10)
        X = X.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)
        model = MLPClassifier().fit(X_train, y_train)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, 'mlp',
            [('input', FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, input_feed={'input': X_test})
        assert_almost_equal(res[1], model.predict_proba(X_test), decimal=5)
        assert_almost_equal(res[0], model.predict(X_test), decimal=5)


if __name__ == "__main__":
    unittest.main()
