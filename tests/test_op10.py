import unittest
from sklearn import datasets
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from onnx.defs import onnx_opset_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml


class TestOp10(unittest.TestCase):
    def _fit_model_binary_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X

    def _fit_model_multiclass_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model, X

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
        model, X = self._fit_model_binary_classification(
            linear_model.LogisticRegression())
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_kmeans(self):
        model, X = self._fit_model_binary_classification(KMeans())
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_gaussian_mixture(self):
        model, X = self._fit_model_binary_classification(
            GaussianMixture())
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_bayesian_gaussian_mixture(self):
        model, X = self._fit_model_binary_classification(
            BayesianGaussianMixture())
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)

    @unittest.skipIf(not onnx_built_with_ml(), reason="onnx-ml")
    @unittest.skipIf(onnx_opset_version() < 10, reason="out of scope")
    def test_gaussian_process_regressor(self):
        model, X = self._fit_model_binary_classification(
            GaussianProcessRegressor())
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
        model, X = self._fit_model_binary_classification(model)
        target_opset = 10
        model_onnx = convert_sklearn(model, "op10",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=target_opset)
        self.check_domain(model_onnx, target_opset=target_opset)


if __name__ == "__main__":
    unittest.main()
