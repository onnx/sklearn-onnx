# SPDX-License-Identifier: Apache-2.0

import unittest
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture
from onnx.defs import onnx_opset_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils.tests_helper import fit_classification_model


class TestOp10(unittest.TestCase):

    def check_domain(self, model, domain="", target_opset=10):
        for op in model.opset_import:
            if op.domain == domain:
                if op.version > target_opset:
                    raise RuntimeError(
                        "Wrong opset {} > {} expected".format(
                            op.domain, target_opset))

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_logistic_regression(self):
        model, X = fit_classification_model(
            linear_model.LogisticRegression(), 3)
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_kmeans(self):
        model, X = fit_classification_model(KMeans(), 3)
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_gaussian_mixture(self):
        model, X = fit_classification_model(GaussianMixture(), 3)
        target_opset = 10
        model_onnx = convert_sklearn(
            model, "op10",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_gaussian_process_regressor(self):
        model, X = fit_classification_model(GaussianProcessRegressor(), 3)
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_voting_classifier(self):
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        model, X = fit_classification_model(model, 3)
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)


if __name__ == "__main__":
    unittest.main()
