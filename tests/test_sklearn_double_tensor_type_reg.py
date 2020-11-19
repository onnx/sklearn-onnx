"""Tests GLMRegressor converter."""

import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
try:
    from sklearn.ensemble import VotingRegressor
except ImportError:
    # New in 0.21
    VotingRegressor = None
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import DoubleTensorType
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_regression_model)  # , TARGET_OPSET)

TARGET_OPSET = 12  # change when PR 551


class TestSklearnDoubleTensorTypeRegressor(unittest.TestCase):
    def test_model_linear_regression_64(self):
        model, X = fit_regression_model(LinearRegression())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(np.float64), model, model_onnx,
            basename="SklearnLinearRegressionDouble")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses implementation for "
               "Relu, Tanh, Sigmoid for double")
    def test_model_mlpregressor_64(self):
        # Could not find an implementation for the node Relu:Relu(6)
        # Could not find an implementation for the node Tanh:Tanh(6)
        # Could not find an implementation for the node Sigmoid:Sigmoid(6)
        for activation in ['relu', 'tanh', 'logistic']:
            with self.subTest(activation=activation):
                model, X = fit_regression_model(
                    MLPRegressor(activation=activation))
                model_onnx = convert_sklearn(
                    model, "linear regression",
                    [("input", DoubleTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET)
                self.assertIn("elem_type: 11", str(model_onnx))
                dump_data_and_model(
                    X.astype(np.float64), model, model_onnx,
                    basename="SklearnMLPRegressorDouble%s" % activation)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses implementation for "
               "ReduceMean for double")
    def test_bagging_regressor_sgd_64(self):
        # Could not find an implementation for
        # the node ReduceMean:ReduceMean(11)
        model, X = fit_regression_model(
            BaggingRegressor(SGDRegressor()))
        model_onnx = convert_sklearn(
            model, "bagging regressor",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(np.float64), model, model_onnx,
            basename="SklearnBaggingRegressorSGDDouble")

    def test_model_sgd_regressor_64(self):
        model, X = fit_regression_model(SGDRegressor())
        model_onnx = convert_sklearn(
            model, "linear regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIn("elem_type: 11", str(model_onnx))
        dump_data_and_model(
            X.astype(np.float64), model, model_onnx,
            basename="SklearnLinearSGDRegressorDouble")

    def test_gpr_rbf_fitted_true_double(self):
        gp = GaussianProcessRegressor(
            alpha=1e-7, n_restarts_optimizer=15, normalize_y=True)
        gp, X = fit_regression_model(gp)
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(np.float64), gp, model_onnx, verbose=False,
            basename="SklearnGaussianProcessRBFTDouble")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses implementation for "
               "TopK for double")
    def test_model_knn_regressor_double(self):
        # Could not find an implementation for the node To_TopK:TopK(11)
        model, X = fit_regression_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(
            model, "KNN regressor", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'optim': 'cdist'}})
        dump_data_and_model(
            X.astype(np.float64)[:7],
            model, model_onnx,
            basename="SklearnKNeighborsRegressorDouble")

    @unittest.skipIf(VotingRegressor is None, reason="new in 0.21")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses implementation for "
               "Sum for double")
    def test_model_voting_regression(self):
        # Could not find an implementation for the node Sum:Sum(8)
        model = VotingRegressor([
            ('lr', LinearRegression()),
            ('dt', SGDRegressor())])
        model, X = fit_regression_model(model)
        model_onnx = convert_sklearn(
            model, "voting regression",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(np.float64), model, model_onnx,
            basename="SklearnVotingRegressorDouble",
            comparable_outputs=[0])


if __name__ == "__main__":
    unittest.main()
