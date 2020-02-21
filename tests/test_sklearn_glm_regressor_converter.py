"""Tests GLMRegressor converter."""

import unittest
from distutils.version import StrictVersion
import numpy
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType
)
from onnxruntime import __version__ as ort_version
from test_utils import dump_data_and_model, fit_regression_model


class TestGLMRegressorConverter(unittest.TestCase):
    def test_model_linear_regression(self):
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
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
        model, X = fit_regression_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(model, "linear regression",
                                     [("input", DoubleTensorType(X.shape))],
                                     dtype=numpy.float64)
        self.assertIsNotNone(model_onnx)
        self.assertIn("elem_type: 11", str(model_onnx))

    def test_model_linear_regression_int(self):
        model, X = fit_regression_model(
            linear_model.LinearRegression(), is_int=True)
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
        model, X = fit_regression_model(
            linear_model.LinearRegression(fit_intercept=False))
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLinearRegressionNoIntercept-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_linear_svr(self):
        model, X = fit_regression_model(LinearSVR())
        model_onnx = convert_sklearn(
            model, "linear SVR",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLinearSvr-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_linear_svr_int(self):
        model, X = fit_regression_model(LinearSVR(), is_int=True)
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
        model, X = fit_regression_model(linear_model.Ridge())
        model_onnx = convert_sklearn(
            model, "ridge regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnRidge-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ridge_int(self):
        model, X = fit_regression_model(linear_model.Ridge(), is_int=True)
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
        model, X = fit_regression_model(linear_model.SGDRegressor())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnSGDRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_sgd_regressor_int(self):
        model, X = fit_regression_model(
            linear_model.SGDRegressor(), is_int=True)
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
        model, X = fit_regression_model(linear_model.ElasticNet())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnElasticNet-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_elastic_net_cv_regressor(self):
        model, X = fit_regression_model(linear_model.ElasticNetCV())
        model_onnx = convert_sklearn(
            model,
            "scikit-learn elastic-net regression",
            [("input", FloatTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnElasticNetCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_elastic_net_regressor_int(self):
        model, X = fit_regression_model(linear_model.ElasticNet(), is_int=True)
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
        model, X = fit_regression_model(linear_model.Lars())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLars-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lars_cv(self):
        model, X = fit_regression_model(linear_model.LarsCV())
        model_onnx = convert_sklearn(
            model, "lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLarsCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars(self):
        model, X = fit_regression_model(linear_model.LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(
            model, "lasso lars",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLassoLars-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars_cv(self):
        model, X = fit_regression_model(linear_model.LassoLarsCV())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLassoLarsCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars_ic(self):
        model, X = fit_regression_model(linear_model.LassoLarsIC())
        model_onnx = convert_sklearn(
            model, "lasso lars cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLassoLarsIC-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_cv(self):
        model, X = fit_regression_model(linear_model.LassoCV())
        model_onnx = convert_sklearn(
            model, "lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnLassoCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_lasso_lars_int(self):
        model, X = fit_regression_model(linear_model.LassoLars(), is_int=True)
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
        model, X = fit_regression_model(linear_model.LinearRegression(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiLinearRegression-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ard_regression(self):
        model, X = fit_regression_model(linear_model.ARDRegression())
        model_onnx = convert_sklearn(
            model, "ard regression",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnARDRegression-Dec3",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_theilsen(self):
        model, X = fit_regression_model(linear_model.TheilSenRegressor())
        model_onnx = convert_sklearn(
            model, "thiel-sen regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnTheilSen-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_bayesian_ridge(self):
        model, X = fit_regression_model(linear_model.BayesianRidge())
        model_onnx = convert_sklearn(
            model, "bayesian ridge",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBayesianRidge-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_huber_regressor(self):
        model, X = fit_regression_model(linear_model.HuberRegressor())
        model_onnx = convert_sklearn(
            model, "huber regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnHuberRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_multi_task_lasso(self):
        model, X = fit_regression_model(linear_model.MultiTaskLasso(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task lasso",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiTaskLasso-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_multi_task_lasso_cv(self):
        model, X = fit_regression_model(linear_model.MultiTaskLassoCV(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "mutli-task lasso cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiTaskLassoCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_multi_task_elasticnet(self):
        model, X = fit_regression_model(linear_model.MultiTaskElasticNet(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task elasticnet",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiTaskElasticNet-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_orthogonal_matching_pursuit(self):
        model, X = fit_regression_model(
            linear_model.OrthogonalMatchingPursuit())
        model_onnx = convert_sklearn(
            model, "orthogonal matching pursuit",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnOrthogonalMatchingPursuit-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_passive_aggressive_regressor(self):
        model, X = fit_regression_model(
            linear_model.PassiveAggressiveRegressor())
        model_onnx = convert_sklearn(
            model, "passive aggressive regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnPassiveAggressiveRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ransac_regressor_default(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor())
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnRANSACRegressor-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ransac_regressor_mlp(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor(
                base_estimator=MLPRegressor(solver='lbfgs')))
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnRANSACRegressorMLP-Dec3",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_ransac_regressor_tree(self):
        model, X = fit_regression_model(
            linear_model.RANSACRegressor(
                base_estimator=GradientBoostingRegressor()))
        model_onnx = convert_sklearn(
            model, "ransac regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnRANSACRegressorTree-Dec3",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_multi_task_elasticnet_cv(self):
        model, X = fit_regression_model(linear_model.MultiTaskElasticNetCV(),
                                        n_targets=2)
        model_onnx = convert_sklearn(
            model, "multi-task elasticnet cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnMultiTaskElasticNetCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    def test_model_orthogonal_matching_pursuit_cv(self):
        model, X = fit_regression_model(
            linear_model.OrthogonalMatchingPursuitCV())
        model_onnx = convert_sklearn(
            model, "orthogonal matching pursuit cv",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnOrthogonalMatchingPursuitCV-Dec4",
            allow_failure="StrictVersion("
            "onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
