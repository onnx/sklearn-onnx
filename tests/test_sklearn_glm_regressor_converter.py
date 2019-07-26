"""Tests GLMRegressor converter."""

import unittest
from distutils.version import StrictVersion
import numpy
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType
)
from onnxruntime import __version__ as ort_version
from test_utils import dump_data_and_model


def _fit_model(model, is_int=False):
    X, y = datasets.make_regression(n_features=4, random_state=0)
    if is_int:
        X = X.astype(numpy.int64)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


def _fit_model_multi(model):
    X, y = datasets.make_regression(n_features=4, random_state=0,
                                    n_targets=2, n_samples=10)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


class TestGLMRegressorConverter(unittest.TestCase):
    def test_model_linear_regression(self):
        model, X = _fit_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLinearRegression-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="old onnxruntime does not support double")
    def test_model_linear_regression64(self):
        model, X = _fit_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", DoubleTensorType(X.shape))],
                                     dtype=numpy.float64)
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))

    def test_model_linear_regression_int(self):
        model, X = _fit_model(linear_model.LinearRegression(), is_int=True)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLinearRegressionInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_linear_regression_nointercept(self):
        model, X = _fit_model(
            linear_model.LinearRegression(fit_intercept=False))
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLinearRegressionNoIntercept-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_linear_svr(self):
        model, X = _fit_model(LinearSVR())
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLinearSvr-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_linear_svr_int(self):
        model, X = _fit_model(LinearSVR(), is_int=True)
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLinearSvrInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ridge(self):
        model, X = _fit_model(linear_model.Ridge())
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnRidge-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ridge_int(self):
        model, X = _fit_model(linear_model.Ridge(), is_int=True)
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidgeInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_sgd_regressor(self):
        model, X = _fit_model(linear_model.SGDRegressor())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnSGDRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_sgd_regressor_int(self):
        model, X = _fit_model(linear_model.SGDRegressor(), is_int=True)
        model_onnx = convert_sklearn(
            model, "SGD regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnSGDRegressorInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_elastic_net_regressor(self):
        model, X = _fit_model(linear_model.ElasticNet())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnElasticNet-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_elastic_net_cv_regressor(self):
        model, X = _fit_model(linear_model.ElasticNetCV())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnElasticNetCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_elastic_net_regressor_int(self):
        model, X = _fit_model(linear_model.ElasticNet(), is_int=True)
        model_onnx = convert_sklearn(
            model, "elastic net regression",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnElasticNetRegressorInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lars(self):
        model, X = _fit_model(linear_model.Lars())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLars-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lars_cv(self):
        model, X = _fit_model(linear_model.LarsCV())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLarsCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars(self):
        model, X = _fit_model(linear_model.LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLassoLars-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars_cv(self):
        model, X = _fit_model(linear_model.LassoLarsCV())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLassoLarsCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars_ic(self):
        model, X = _fit_model(linear_model.LassoLarsIC())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLassoLarsIC-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_cv(self):
        model, X = _fit_model(linear_model.LassoCV())
        model_onnx = convert_sklearn(
            model, "lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnLassoCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars_int(self):
        model, X = _fit_model(linear_model.LassoLars(), is_int=True)
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", Int64TensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLassoLarsInt-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_multi_linear_regression(self):
        model, X = _fit_model_multi(linear_model.LinearRegression())
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", FloatTensorType([None, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            verbose=False,
            basename="SklearnLinearRegression-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
