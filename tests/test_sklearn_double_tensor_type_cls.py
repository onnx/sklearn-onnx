"""Tests GLMRegressor converter."""

import unittest
from distutils.version import StrictVersion
import numpy as np
# from sklearn.ensemble import BaggingClassifier
# Requires PR #488.
# from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression  # , SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
try:
    from sklearn.ensemble import VotingClassifier
except ImportError:
    # New in 0.21
    VotingClassifier = None
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_classification_model)  # , TARGET_OPSET)

TARGET_OPSET = 12  # change when PR 551


class TestSklearnDoubleTensorTypeClassifier(unittest.TestCase):
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.5.0"),
        reason="ArgMax is missing")
    @ignore_warnings(category=DeprecationWarning)
    def test_model_logistic_regression_64_binary(self):
        for n_cl in [2, 3]:
            model, X = fit_classification_model(
                LogisticRegression(), n_cl, n_features=4)
            pmethod = ('decision_function_binary' if n_cl == 2 else
                       'decision_function')
            for b in [True, False]:
                for z in [False]:
                    # zipmap does not allow tensor(double) as inputs
                    with self.subTest(n_classes=n_cl, raw_score=b):
                        model_onnx = convert_sklearn(
                            model, "model",
                            [("input", DoubleTensorType([None, X.shape[1]]))],
                            target_opset=TARGET_OPSET,
                            options={id(model): {"raw_scores": b,
                                                 "zipmap": z}})
                        self.assertIn("elem_type: 11", str(model_onnx))
                        methods = None if not b else ['predict', pmethod]
                        if not b and n_cl == 2:
                            # onnxruntime does not support sigmoid for
                            # DoubleTensorType
                            continue
                        dump_data_and_model(
                            X.astype(np.float64), model, model_onnx,
                            methods=methods,
                            basename="SklearnLogisticRegressionDouble2RAW{}"
                                     "ZIP{}CL{}".format(
                                        1 if b else 0, 1 if z else 0, n_cl))

    """
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

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("1.2.0"),
        reason="onnxruntime misses implementation for double")
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

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="shape_inference fails")
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
    """


if __name__ == "__main__":
    unittest.main()
