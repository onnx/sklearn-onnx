from distutils.version import StrictVersion
import unittest
import numpy as np
import onnx
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, fit_classification_model


class TestNaiveBayesConverter(unittest.TestCase):

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_binary_classification(self):
        model, X = fit_classification_model(
            MultinomialNB(), 2, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32),
            model,
            model_onnx,
            basename="SklearnBinMultinomialNB-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_binary_classification_double(self):
        model, X = fit_classification_model(
            MultinomialNB(), 2, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            dtype=np.float64,
        )
        self.assertIsNotNone(model_onnx)
        # runtime does not allow double in ZipMap
        dump_data_and_model(
            X.astype(np.float64),
            model,
            model_onnx,
            basename="SklearnBinMultinomialNBDouble-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.5.0')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.3"),
        reason="Requires opset 9.",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_bernoulli_nb_binary_classification(self):
        model, X = fit_classification_model(
            BernoulliNB(), 2)
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
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
        model, X = fit_classification_model(
            MultinomialNB(), 5, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclMultinomialNB-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_multiclass_params(self):
        model, X = fit_classification_model(
            MultinomialNB(alpha=0.5, fit_prior=False), 5, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclMultinomialNBParams-Dec4",
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
        model, X = fit_classification_model(
            BernoulliNB(), 4)
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
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

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.3"),
        reason="Requires opset 9.",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_bernoulli_nb_multiclass_params(self):
        model, X = fit_classification_model(
            BernoulliNB(alpha=0, binarize=1.0, fit_prior=False), 4)
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclBernoulliNBParams",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            MultinomialNB(), 2, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinMultinomialNBInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.3"),
        reason="Requires opset 9.",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_bernoulli_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            BernoulliNB(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinBernoulliNBInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multinomial_nb_multiclass_int(self):
        model, X = fit_classification_model(
            MultinomialNB(), 5, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "multinomial naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclMultinomialNBInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) <= StrictVersion("1.3"),
        reason="Requires opset 9.",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_bernoulli_nb_multiclass_int(self):
        model, X = fit_classification_model(
            BernoulliNB(), 4, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "bernoulli naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclBernoulliNBInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_gaussian_nb_binary_classification(self):
        model, X = fit_classification_model(
            GaussianNB(), 2)
        model_onnx = convert_sklearn(
            model,
            "gaussian naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinGaussianNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_gaussian_nb_multiclass(self):
        model, X = fit_classification_model(
            GaussianNB(), 4)
        model_onnx = convert_sklearn(
            model,
            "gaussian naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclGaussianNB",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_gaussian_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            GaussianNB(), 2, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "gaussian naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinGaussianNBInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_gaussian_nb_multiclass_int(self):
        model, X = fit_classification_model(
            GaussianNB(), 5, is_int=True)
        model_onnx = convert_sklearn(
            model,
            "gaussian naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclGaussianNBInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_complement_nb_binary_classification(self):
        model, X = fit_classification_model(
            ComplementNB(), 2, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "complement naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinComplementNB-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_complement_nb_multiclass(self):
        model, X = fit_classification_model(
            ComplementNB(), 4, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "complement naive bayes",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclComplementNB-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_complement_nb_binary_classification_int(self):
        model, X = fit_classification_model(
            ComplementNB(), 2, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "complement naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinComplementNBInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_complement_nb_multiclass_int(self):
        model, X = fit_classification_model(
            ComplementNB(), 5, is_int=True, pos_features=True)
        model_onnx = convert_sklearn(
            model,
            "complement naive bayes",
            [("input", Int64TensorType([None, X.shape[1]]))],
            dtype=np.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclComplementNBInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
