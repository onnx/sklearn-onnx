# SPDX-License-Identifier: Apache-2.0

"""Tests GLMRegressor converter."""

import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model
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
from onnxruntime import __version__ as ort_version, InferenceSession
from test_utils import (
    dump_data_and_model, fit_regression_model, TARGET_OPSET)


ort_version = ort_version.split('+')[0]


class TestGLMRegressorConverter(unittest.TestCase):
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegression-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_blacklist(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            black_op={'LinearRegressor'})
        self.assertNotIn('LinearRegressor', str(model_onnx))
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionBlackOp-Dec4")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.5.0"),
        reason="old onnxruntime does not support double")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_multi(self):
        model, X = fit_regression_model(linear_model.LinearRegression(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionMulti-Dec4")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.5.0"),
        reason="old onnxruntime does not support double")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression64(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", DoubleTensorType(X.shape))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(numpy.float64), model, model_onnx,
            basename="SklearnLinearRegression64-Dec4")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.5.0"),
        reason="old onnxruntime does not support double")
    def test_model_linear_regression64_multiple(self):
        model, X = fit_regression_model(linear_model.LinearRegression(),
                                        n_targets=2)
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", DoubleTensorType(X.shape))],
                                     target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(numpy.float64), model, model_onnx,
            basename="SklearnLinearRegression64Multi-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_int(self):
        model, X = fit_regression_model(
            linear_model.LinearRegression(), is_int=True)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_nointercept(self):
        model, X = fit_regression_model(
            linear_model.LinearRegression(fit_intercept=False))
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionNoIntercept-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_regression_bool(self):
        model, X = fit_regression_model(
            linear_model.LinearRegression(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearRegressionBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_svr(self):
        model, X = fit_regression_model(LinearSVR())
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSvr-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_svr_int(self):
        model, X = fit_regression_model(LinearSVR(), is_int=True)
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSvrInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_linear_svr_bool(self):
        model, X = fit_regression_model(LinearSVR(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLinearSVRBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ridge(self):
        model, X = fit_regression_model(linear_model.Ridge())
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidge-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ridge_int(self):
        model, X = fit_regression_model(linear_model.Ridge(), is_int=True)
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ridge_bool(self):
        model, X = fit_regression_model(linear_model.Ridge(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnRidgeBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_sgd_regressor(self):
        model, X = fit_regression_model(linear_model.SGDRegressor())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnSGDRegressor-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_sgd_regressor_int(self):
        model, X = fit_regression_model(
            linear_model.SGDRegressor(), is_int=True)
        model_onnx = convert_sklearn(
            model, "SGD regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnSGDRegressorInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_sgd_regressor_bool(self):
        model, X = fit_regression_model(
            linear_model.SGDRegressor(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "SGD regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSGDRegressorBool-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_regressor(self):
        model, X = fit_regression_model(linear_model.ElasticNet())
        model_onnx = convert_sklearn(
            model, "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNet-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_cv_regressor(self):
        model, X = fit_regression_model(linear_model.ElasticNetCV())
        model_onnx = convert_sklearn(
            model, "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNetCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_regressor_int(self):
        model, X = fit_regression_model(linear_model.ElasticNet(), is_int=True)
        model_onnx = convert_sklearn(
            model, "elastic net regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNetRegressorInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_elastic_net_regressor_bool(self):
        model, X = fit_regression_model(
            linear_model.ElasticNet(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "elastic net regression",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnElasticNetRegressorBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lars(self):
        model, X = fit_regression_model(linear_model.Lars())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, basename="SklearnLars-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lars_cv(self):
        model, X = fit_regression_model(linear_model.LarsCV())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLarsCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars(self):
        model, X = fit_regression_model(linear_model.LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLars-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_cv(self):
        model, X = fit_regression_model(linear_model.LassoLarsCV())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_ic(self):
        model, X = fit_regression_model(linear_model.LassoLarsIC())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsIC-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_cv(self):
        model, X = fit_regression_model(linear_model.LassoCV())
        model_onnx = convert_sklearn(
            model, "lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_int(self):
        model, X = fit_regression_model(linear_model.LassoLars(), is_int=True)
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsInt-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_lasso_lars_bool(self):
        model, X = fit_regression_model(
            linear_model.LassoLars(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnLassoLarsBool")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_linear_regression(self):
        model, X = fit_regression_model(linear_model.LinearRegression(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiLinearRegression-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ard_regression(self):
        model, X = fit_regression_model(
            linear_model.ARDRegression(), factor=0.001)
        model_onnx = convert_sklearn(
            model, "ard regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnARDRegression-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_theilsen(self):
        model, X = fit_regression_model(linear_model.TheilSenRegressor())
        model_onnx = convert_sklearn(
            model, "thiel-sen regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnTheilSen-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge(self):
        model, X = fit_regression_model(linear_model.BayesianRidge())
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBayesianRidge-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std(self):
        model, X = fit_regression_model(linear_model.BayesianRidge(),
                                        n_features=2, n_samples=20)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {'return_std': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)

        sess = InferenceSession(model_onnx.SerializeToString())
        outputs = sess.run(None, {'input': X})
        pred, std = model.predict(X, return_std=True)
        assert_almost_equal(pred, outputs[0].ravel(), decimal=4)
        assert_almost_equal(std, outputs[1].ravel(), decimal=4)

    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.3.0"),
                     reason="output type")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std_double(self):
        model, X = fit_regression_model(linear_model.BayesianRidge(),
                                        n_features=2, n_samples=100,
                                        n_informative=1)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {'return_std': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)

        X = X.astype(numpy.float64)
        sess = InferenceSession(model_onnx.SerializeToString())
        outputs = sess.run(None, {'input': X})
        pred, std = model.predict(X, return_std=True)
        assert_almost_equal(pred, outputs[0].ravel())
        assert_almost_equal(std, outputs[1].ravel(), decimal=4)

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std_normalize(self):
        model, X = fit_regression_model(
            linear_model.BayesianRidge(normalize=True),
            n_features=2, n_samples=50)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {'return_std': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)

        sess = InferenceSession(model_onnx.SerializeToString())
        outputs = sess.run(None, {'input': X})
        pred, std = model.predict(X, return_std=True)
        assert_almost_equal(pred, outputs[0].ravel(), decimal=4)
        assert_almost_equal(std, outputs[1].ravel(), decimal=4)

    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.3.0"),
                     reason="output type")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_bayesian_ridge_return_std_normalize_double(self):
        model, X = fit_regression_model(
            linear_model.BayesianRidge(normalize=True),
            n_features=2, n_samples=50)
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            options={linear_model.BayesianRidge: {'return_std': True}},
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)

        X = X.astype(numpy.float64)
        sess = InferenceSession(model_onnx.SerializeToString())
        outputs = sess.run(None, {'input': X})
        pred, std = model.predict(X, return_std=True)
        assert_almost_equal(pred, outputs[0].ravel())
        assert_almost_equal(std, outputs[1].ravel(), decimal=4)

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_huber_regressor(self):
        model, X = fit_regression_model(linear_model.HuberRegressor())
        model_onnx = convert_sklearn(
            model, "huber regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnHuberRegressor-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_lasso(self):
        model, X = fit_regression_model(linear_model.MultiTaskLasso(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task lasso",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskLasso-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_lasso_cv(self):
        model, X = fit_regression_model(linear_model.MultiTaskLassoCV(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "mutli-task lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskLassoCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_elasticnet(self):
        model, X = fit_regression_model(linear_model.MultiTaskElasticNet(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task elasticnet",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskElasticNet-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_orthogonal_matching_pursuit(self):
        model, X = fit_regression_model(
            linear_model.OrthogonalMatchingPursuit())
        model_onnx = convert_sklearn(
            model, "orthogonal matching pursuit",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnOrthogonalMatchingPursuit-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_passive_aggressive_regressor(self):
        model, X = fit_regression_model(
            linear_model.PassiveAggressiveRegressor())
        model_onnx = convert_sklearn(
            model, "passive aggressive regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnPassiveAggressiveRegressor-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ransac_regressor_default(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor())
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnRANSACRegressor-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ransac_regressor_mlp(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor(
                base_estimator=MLPRegressor(solver='sgd', max_iter=20)))
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnRANSACRegressorMLP-Dec3")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_ransac_regressor_tree(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor(
                base_estimator=GradientBoostingRegressor()))
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnRANSACRegressorTree-Dec3")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning))
    def test_model_multi_task_elasticnet_cv(self):
        model, X = fit_regression_model(linear_model.MultiTaskElasticNetCV(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task elasticnet cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnMultiTaskElasticNetCV-Dec4")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_model_orthogonal_matching_pursuit_cv(self):
        model, X = fit_regression_model(
            linear_model.OrthogonalMatchingPursuitCV())
        model_onnx = convert_sklearn(
            model, "orthogonal matching pursuit cv",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx, verbose=False,
            basename="SklearnOrthogonalMatchingPursuitCV-Dec4")

    def check_model(self, model, X, name='input'):
        try:
            sess = InferenceSession(model.SerializeToString())
        except Exception as e:
            raise AssertionError(
                "Unable to load model\n%s" % str(model)) from e
        try:
            return sess.run(None, {name: X[:7]})
        except Exception as e:
            raise AssertionError(
                "Unable to run model X.shape=%r X.dtype=%r\n%s" % (
                    X[:7].shape, X.dtype, str(model))) from e

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    @unittest.skipIf(PoissonRegressor is None,
                     reason="scikit-learn too old")
    def test_model_poisson_regressor(self):
        X, y = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42,
            n_informative=3)
        y = numpy.abs(y)
        y = y / y.max() + 1e-5
        model = linear_model.PoissonRegressor().fit(X, y)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.check_model(model_onnx, X.astype(numpy.float32))
        dump_data_and_model(
            X.astype(numpy.float32), model, model_onnx,
            basename="SklearnPoissonRegressor-Dec4")
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float64), model, model_onnx,
            basename="SklearnPoissonRegressor64")

    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    @unittest.skipIf(TweedieRegressor is None,
                     reason="scikti-learn too old")
    def test_model_tweedie_regressor(self):
        X, y = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42,
            n_informative=3)
        y = numpy.abs(y)
        y = y / y.max() + 1e-5
        for power in range(0, 4):
            with self.subTest(power=power):
                model = linear_model.TweedieRegressor(power=power).fit(X, y)
                model_onnx = convert_sklearn(
                    model, "linear regression",
                    [("input", FloatTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET)
                self.check_model(model_onnx, X.astype(numpy.float32))
                dump_data_and_model(
                    X.astype(numpy.float32), model, model_onnx,
                    basename="SklearnTweedieRegressor%d-Dec4" % power)
                model_onnx = convert_sklearn(
                    model, "linear regression",
                    [("input", DoubleTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET)
                dump_data_and_model(
                    X.astype(numpy.float64), model, model_onnx,
                    basename="SklearnTweedieRegressor64%d" % power)

    @unittest.skipIf(QuantileRegressor is None,
                     reason="scikit-learn<1.0")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_model_quantile_regressor(self):
        X, y = make_regression(
            n_features=5, n_samples=100, n_targets=1, random_state=42,
            n_informative=3)
        y = numpy.abs(y)
        y = y / y.max() + 1e-5
        model = linear_model.QuantileRegressor().fit(X, y)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.check_model(model_onnx, X.astype(numpy.float32))
        dump_data_and_model(
            X.astype(numpy.float32), model, model_onnx,
            basename="SklearnQuantileRegressor-Dec4")
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float64), model, model_onnx,
            basename="SklearnQuantileRegressor64")


if __name__ == "__main__":
    unittest.main()
