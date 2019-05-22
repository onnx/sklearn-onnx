import numpy as np
import onnx
import unittest
from distutils.version import StrictVersion
from sklearn.datasets import load_digits, load_iris
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


class TestNaiveBayesConverter(unittest.TestCase):
    def _fit_model_binary_classification(self, model, data):
        X = data.data
        y = data.target
        mid_point = len(data.target_names) / 2
        y[y < mid_point] = 0
        y[y >= mid_point] = 1
        model.fit(X, y)
        return model, X.astype(np.float32)

    def _fit_model_multiclass_classification(self, model, data):
        X = data.data
        y = data.target
        model.fit(X, y)
        return model, X.astype(np.float32)

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_binary_classification(self):
        model, X = self._fit_model_binary_classification(
            MultinomialNB(), load_iris())
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinMultinomialNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.3"),
        reason="Requires opset 9.",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_bernoulli_nb_binary_classification(self):
        model, X = self._fit_model_binary_classification(
            BernoulliNB(), load_digits())
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinBernoulliNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_multiclass(self):
        model, X = self._fit_model_multiclass_classification(
            MultinomialNB(), load_iris())
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclMultinomialNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.3"),
        reason="Requires opset 9.",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_bernoulli_nb_multiclass(self):
        model, X = self._fit_model_multiclass_classification(
            BernoulliNB(), load_digits())
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", FloatTensorType(X.shape))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclBernoulliNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
