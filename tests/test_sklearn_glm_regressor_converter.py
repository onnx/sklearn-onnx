# SPDX-License-Identifier: Apache-2.0

"""Tests GLMRegressor converter."""

import unittest
import packaging.version as pv
import onnx
import numpy
from numpy.testing import assert_allclose, assert_almost_equal

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model, __version__ as sklearn_version
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

try:
    from sklearn.linear_model import QuantileRegressor
except (ImportError, AttributeError):
    # available since sklearn>=1.0
    QuantileRegressor = None
try:
    from sklearn.linear_model import PoissonRegressor
except (ImportError, AttributeError):
    # available since sklearn>=0.23
    PoissonRegressor = None
try:
    from sklearn.linear_model import TweedieRegressor
except (ImportError, AttributeError):
    # available since sklearn>=0.23
    TweedieRegressor = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int64TensorType,
)
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model,
    fit_regression_model,
    TARGET_OPSET,
    InferenceSessionEx as InferenceSession,
)

ort_version = ort_version.split("+")[0]
skl_version = ".".join(sklearn_version.split(".")[:2])

BACKEND = (
    "onnxruntime"
    if pv.Version(onnx.__version__) < pv.Version("1.16.0")
    else "onnx;onnxruntime"
)


def _fit_bayesian_ridge_std_model(dtype):
    X_train = numpy.array(
        [
            [8.2, -3.4, 5.1, 12.0],
            [9.7, -1.2, 7.4, 10.5],
            [11.3, -4.8, 6.2, 14.1],
            [7.1, -2.5, 9.0, 11.7],
            [12.8, -0.7, 4.3, 15.6],
            [10.4, -5.1, 8.5, 9.4],
            [6.6, -1.8, 3.7, 13.3],
            [13.5, -3.9, 7.8, 16.2],
            [8.9, -6.0, 5.6, 10.1],
            [14.2, -2.2, 9.3, 12.8],
            [9.1, -4.1, 4.8, 17.0],
            [11.8, -1.5, 8.1, 8.7],
        ],
        dtype=dtype,
    )
    y_train = numpy.array(
        [18.1, 25.4, 20.2, 28.7, 21.9, 31.3, 12.8, 27.6, 22.4, 35.1, 16.7, 30.8],
        dtype=dtype,
    )
    X_test = numpy.array(
        [
            [7.8, -2.9, 6.7, 14.8],
            [12.1, -5.4, 3.9, 9.8],
            [15.0, -0.9, 8.8, 18.1],
            [9.6, -3.7, 10.2, 11.3],
        ],
        dtype=dtype,
    )
    model = linear_model.BayesianRidge().fit(X_train, y_train)
    return model, X_test


def _bayesian_ridge_variance_input(model, X):
    sklearn_release = pv.Version(sklearn_version).release[:2]
    if sklearn_release >= (1, 9):
        return X - model.X_offset_
    if sklearn_release < (1, 2) and getattr(model, "_normalize", False):
        return (X - model.X_offset_) / model.X_scale_
    return X


def _bayesian_ridge_std_reference(model, X):
    variance_input = _bayesian_ridge_variance_input(model, X)
    projected = numpy.matmul(variance_input, model.sigma_)
    quadratic = numpy.sum(projected * variance_input, axis=1)
    return numpy.sqrt(quadratic + 1.0 / model.alpha_)


class TestGLMRegressorConverter(unittest.TestCase):
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearRegression-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_blacklist(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            black_op={"LinearRegressor"},
        )
        self.assertNotIn("LinearRegressor", str(model_onnx))
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearRegressionBlackOp-Dec4"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.5.0"),
        reason="old onnxruntime does not support double",
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_multi(self):
        model, X = fit_regression_model(linear_model.LinearRegression(), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearRegressionMulti-Dec4"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.5.0"),
        reason="old onnxruntime does not support double",
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression64(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", DoubleTensorType(X.shape))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(numpy.float64),
            model,
            model_onnx,
            basename="SklearnLinearRegression64-Dec4",
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.5.0"),
        reason="old onnxruntime does not support double",
    )
    def test_model_linear_regression64_multiple(self):
        model, X = fit_regression_model(linear_model.LinearRegression(), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", DoubleTensorType(X.shape))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(numpy.float64),
            model,
            model_onnx,
            basename="SklearnLinearRegression64Multi-Dec4",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_int(self):
        model, X = fit_regression_model(linear_model.LinearRegression(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearRegressionInt-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_nointercept(self):
        model, X = fit_regression_model(
            linear_model.LinearRegression(fit_intercept=False)
        )
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearRegressionNoIntercept-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_bool(self):
        model, X = fit_regression_model(linear_model.LinearRegression(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLinearRegressionBool"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_svr(self):
        model, X = fit_regression_model(LinearSVR())
        model_onnx = convert_sklearn(
            model,
            "linear SVR",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLinearSvr-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_svr_int(self):
        model, X = fit_regression_model(LinearSVR(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "linear SVR",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLinearSvrInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_svr_bool(self):
        model, X = fit_regression_model(LinearSVR(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "linear SVR",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLinearSVRBool")

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge(self):
        model, X = fit_regression_model(linear_model.Ridge())
        model_onnx = convert_sklearn(
            model,
            "ridge regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRidge-Dec4")

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_int(self):
        model, X = fit_regression_model(linear_model.Ridge(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "ridge regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRidgeInt-Dec4")

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("1.11.0"),
        reason="onnxruntime not recent enough",
    )
    @unittest.skipIf(
        pv.Version(skl_version) <= pv.Version("1.1.0"),
        reason="sklearn fails on windows",
    )
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_model_ridge_bool(self):
        model, X = fit_regression_model(linear_model.Ridge(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "ridge regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRidgeBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_sgd_regressor(self):
        model, X = fit_regression_model(linear_model.SGDRegressor())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnSGDRegressor-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_sgd_regressor_int(self):
        model, X = fit_regression_model(linear_model.SGDRegressor(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "SGD regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnSGDRegressorInt-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_sgd_regressor_bool(self):
        model, X = fit_regression_model(linear_model.SGDRegressor(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "SGD regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnSGDRegressorBool-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_regressor(self):
        model, X = fit_regression_model(linear_model.ElasticNet())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnElasticNet-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_cv_regressor(self):
        model, X = fit_regression_model(linear_model.ElasticNetCV())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnElasticNetCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_regressor_int(self):
        model, X = fit_regression_model(linear_model.ElasticNet(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "elastic net regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnElasticNetRegressorInt-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_regressor_bool(self):
        model, X = fit_regression_model(linear_model.ElasticNet(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "elastic net regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnElasticNetRegressorBool"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lars(self):
        model, X = fit_regression_model(linear_model.Lars())
        model_onnx = convert_sklearn(
            model,
            "lars",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLars-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lars_cv(self):
        model, X = fit_regression_model(linear_model.LarsCV())
        model_onnx = convert_sklearn(
            model,
            "lars",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLarsCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars(self):
        model, X = fit_regression_model(linear_model.LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(
            model,
            "lasso lars",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLassoLars-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_cv(self):
        model, X = fit_regression_model(linear_model.LassoLarsCV())
        model_onnx = convert_sklearn(
            model,
            "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLassoLarsCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_ic(self):
        model, X = fit_regression_model(linear_model.LassoLarsIC())
        model_onnx = convert_sklearn(
            model,
            "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLassoLarsIC-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_cv(self):
        model, X = fit_regression_model(linear_model.LassoCV())
        model_onnx = convert_sklearn(
            model,
            "lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLassoCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_int(self):
        model, X = fit_regression_model(linear_model.LassoLars(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "lasso lars",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLassoLarsInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_bool(self):
        model, X = fit_regression_model(linear_model.LassoLars(), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "lasso lars",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnLassoLarsBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_linear_regression(self):
        model, X = fit_regression_model(linear_model.LinearRegression(), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiLinearRegression-Dec4",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ard_regression(self):
        model, X = fit_regression_model(linear_model.ARDRegression(), factor=0.001)
        model_onnx = convert_sklearn(
            model,
            "ard regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnARDRegression-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_theilsen(self):
        model, X = fit_regression_model(linear_model.TheilSenRegressor())
        model_onnx = convert_sklearn(
            model,
            "thiel-sen regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnTheilSen-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge(self):
        model, X = fit_regression_model(linear_model.BayesianRidge())
        model_onnx = convert_sklearn(
            model,
            "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnBayesianRidge-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std(self):
        model, X = _fit_bayesian_ridge_std_model(numpy.float32)
        self.assertTrue(numpy.all(numpy.abs(model.X_offset_) > 1.0))

        sklearn_mean, sklearn_std = model.predict(X, return_std=True)
        reference_std = _bayesian_ridge_std_reference(model, X)
        assert_allclose(sklearn_std, reference_std, rtol=1e-6, atol=1e-6)

        variance_input = _bayesian_ridge_variance_input(model, X)
        projected = numpy.matmul(variance_input, model.sigma_)
        quadratic = numpy.sum(projected * variance_input, axis=1)
        missing_hadamard = numpy.sum(projected, axis=1)
        self.assertFalse(numpy.allclose(quadratic, missing_hadamard))
        if pv.Version(sklearn_version).release[:2] >= (1, 9):
            raw_projected = numpy.matmul(X, model.sigma_)
            raw_quadratic = numpy.sum(raw_projected * X, axis=1)
            self.assertFalse(numpy.allclose(quadratic, raw_quadratic))

        for target_opset in (12, 13):
            with self.subTest(target_opset=target_opset):
                model_onnx = convert_sklearn(
                    model,
                    "bayesian ridge",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    options={linear_model.BayesianRidge: {"return_std": True}},
                    target_opset=target_opset,
                )
                self.assertIsNotNone(model_onnx)
                self.assertIn("Mul", {node.op_type for node in model_onnx.graph.node})

                std_dims = model_onnx.graph.output[1].type.tensor_type.shape.dim
                self.assertEqual(len(std_dims), 2)
                self.assertEqual(std_dims[1].dim_value, 1)

                sess = InferenceSession(
                    model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                onnx_mean, onnx_std = sess.run(None, {"input": X})
                self.assertEqual(onnx_std.shape, (X.shape[0], 1))
                assert_allclose(sklearn_mean, onnx_mean.ravel(), rtol=1e-5, atol=1e-5)
                assert_allclose(sklearn_std, onnx_std.ravel(), rtol=1e-5, atol=1e-5)
                assert_allclose(reference_std, onnx_std.ravel(), rtol=1e-5, atol=1e-5)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.3.0"), reason="output type"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std_double(self):
        model, X = _fit_bayesian_ridge_std_model(numpy.float64)
        model_onnx = convert_sklearn(
            model,
            "bayesian ridge",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {"return_std": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        onnx_mean, onnx_std = sess.run(None, {"input": X})
        sklearn_mean, sklearn_std = model.predict(X, return_std=True)
        reference_std = _bayesian_ridge_std_reference(model, X)
        assert_allclose(sklearn_std, reference_std, rtol=1e-9, atol=1e-9)
        assert_allclose(sklearn_mean, onnx_mean.ravel(), rtol=1e-7, atol=1e-7)
        assert_allclose(sklearn_std, onnx_std.ravel(), rtol=1e-7, atol=1e-7)

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std_normalize(self):
        try:
            model = linear_model.BayesianRidge(normalize=True)
        except TypeError:
            # normalize not supported anymore
            return
        model, X = fit_regression_model(model, n_features=2, n_samples=50)
        model_onnx = convert_sklearn(
            model,
            "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {"return_std": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outputs = sess.run(None, {"input": X})
        pred, std = model.predict(X, return_std=True)
        assert_almost_equal(pred, outputs[0].ravel(), decimal=4)
        assert_almost_equal(std, outputs[1].ravel(), decimal=4)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.3.0"), reason="output type"
    )
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std_normalize_double(self):
        try:
            model = linear_model.BayesianRidge(normalize=True)
        except TypeError:
            # normalize not supported anymore
            return
        model, X = fit_regression_model(model, n_features=2, n_samples=50)
        model_onnx = convert_sklearn(
            model,
            "bayesian ridge",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {"return_std": True}},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)

        X = X.astype(numpy.float64)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outputs = sess.run(None, {"input": X})
        pred, std = model.predict(X, return_std=True)
        assert_almost_equal(pred, outputs[0].ravel())
        assert_almost_equal(std, outputs[1].ravel(), decimal=4)

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_huber_regressor(self):
        model, X = fit_regression_model(linear_model.HuberRegressor())
        model_onnx = convert_sklearn(
            model,
            "huber regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnHuberRegressor-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_lasso(self):
        model, X = fit_regression_model(linear_model.MultiTaskLasso(), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "multi-task lasso",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False, basename="SklearnMultiTaskLasso-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_lasso_cv(self):
        model, X = fit_regression_model(linear_model.MultiTaskLassoCV(), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "mutli-task lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False, basename="SklearnMultiTaskLassoCV-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_elasticnet(self):
        model, X = fit_regression_model(linear_model.MultiTaskElasticNet(), n_targets=2)
        model_onnx = convert_sklearn(
            model,
            "multi-task elasticnet",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiTaskElasticNet-Dec4",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_orthogonal_matching_pursuit(self):
        model, X = fit_regression_model(linear_model.OrthogonalMatchingPursuit())
        model_onnx = convert_sklearn(
            model,
            "orthogonal matching pursuit",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnOrthogonalMatchingPursuit-Dec4",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_passive_aggressive_regressor(self):
        model, X = fit_regression_model(linear_model.PassiveAggressiveRegressor())
        model_onnx = convert_sklearn(
            model,
            "passive aggressive regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnPassiveAggressiveRegressor-Dec4",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ransac_regressor_default(self):
        model, X = fit_regression_model(linear_model.RANSACRegressor())
        model_onnx = convert_sklearn(
            model,
            "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False, basename="SklearnRANSACRegressor-Dec4"
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ransac_regressor_mlp(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor(
                MLPRegressor(solver="sgd", max_iter=20), min_samples=5
            )
        )
        model_onnx = convert_sklearn(
            model,
            "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnRANSACRegressorMLP-Dec3",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ransac_regressor_tree(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor(GradientBoostingRegressor(), min_samples=5),
            n_features=5,
            n_samples=100,
        )
        model_onnx = convert_sklearn(
            model,
            "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnRANSACRegressorTree-Dec3",
            backend=BACKEND,
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_elasticnet_cv(self):
        model, X = fit_regression_model(
            linear_model.MultiTaskElasticNetCV(), n_targets=2
        )
        model_onnx = convert_sklearn(
            model,
            "multi-task elasticnet cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiTaskElasticNetCV-Dec4",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_orthogonal_matching_pursuit_cv(self):
        model, X = fit_regression_model(linear_model.OrthogonalMatchingPursuitCV())
        model_onnx = convert_sklearn(
            model,
            "orthogonal matching pursuit cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnOrthogonalMatchingPursuitCV-Dec4",
        )

    def check_model(self, model, X, name="input"):
        try:
            sess = InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise AssertionError("Unable to load model\n%s" % str(model)) from e
        try:
            return sess.run(None, {name: X[:7]})
        except Exception as e:
            raise AssertionError(
                "Unable to run model X.shape=%r X.dtype=%r\n%s"
                % (X[:7].shape, X.dtype, str(model))
            ) from e

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(PoissonRegressor is None, reason="scikit-learn too old")
    def test_model_poisson_regressor(self):
        X, y = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42, n_informative=3
        )
        y = numpy.abs(y)
        y = y / y.max() + 1e-5
        model = linear_model.PoissonRegressor().fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.check_model(model_onnx, X.astype(numpy.float32))
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnPoissonRegressor-Dec4",
        )
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float64),
            model,
            model_onnx,
            basename="SklearnPoissonRegressor64",
        )

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    @unittest.skipIf(TweedieRegressor is None, reason="scikti-learn too old")
    def test_model_tweedie_regressor(self):
        X, y = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42, n_informative=3
        )
        y = numpy.abs(y)
        y = y / y.max() + 1e-5
        for power in range(0, 4):
            with self.subTest(power=power):
                model = linear_model.TweedieRegressor(power=power).fit(X, y)
                model_onnx = convert_sklearn(
                    model,
                    "linear regression",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET,
                )
                self.check_model(model_onnx, X.astype(numpy.float32))
                dump_data_and_model(
                    X.astype(numpy.float32),
                    model,
                    model_onnx,
                    basename="SklearnTweedieRegressor%d-Dec4" % power,
                )
                model_onnx = convert_sklearn(
                    model,
                    "linear regression",
                    [("input", DoubleTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET,
                )
                dump_data_and_model(
                    X.astype(numpy.float64),
                    model,
                    model_onnx,
                    basename="SklearnTweedieRegressor64%d" % power,
                )

    @unittest.skipIf(QuantileRegressor is None, reason="scikit-learn<1.0")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning, DeprecationWarning))
    def test_model_quantile_regressor(self):
        X, y = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42, n_informative=3
        )
        y = numpy.abs(y)
        y = y / y.max() + 1e-5
        model = linear_model.QuantileRegressor(solver="highs").fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.check_model(model_onnx, X.astype(numpy.float32))
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnQuantileRegressor-Dec4",
        )
        model_onnx = convert_sklearn(
            model,
            "linear regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            X.astype(numpy.float64),
            model,
            model_onnx,
            basename="SklearnQuantileRegressor64",
        )


if __name__ == "__main__":
    unittest.main()
