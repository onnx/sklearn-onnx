# SPDX-License-Identifier: Apache-2.0


import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import convert_sklearn, parse_sklearn_submodel
from test_utils import TARGET_OPSET, InferenceSessionEx as InferenceSession


class TestParsingOptions(unittest.TestCase):
    def test_pipeline(self):
        model = Pipeline([("sc1", StandardScaler()), ("sc2", StandardScaler())])
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X)
        initial_types = [("input", FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types, target_opset=TARGET_OPSET
        )
        assert model_onnx is not None
        model_onnx = convert_sklearn(
            model,
            initial_types=initial_types,
            final_types=[("output", None)],
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        assert sess.get_outputs()[0].name == "output"
        model_onnx = convert_sklearn(
            model,
            initial_types=initial_types,
            final_types=[("output4", None)],
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        assert sess.get_outputs()[0].name == "output4"
        model_onnx = convert_sklearn(
            model,
            initial_types=initial_types,
            final_types=[("output4", DoubleTensorType())],
            target_opset=TARGET_OPSET,
        )
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except RuntimeError as e:
            if "Cast(9)" in str(e):
                return
            raise e
        assert sess.get_outputs()[0].name == "output4"
        assert str(sess.get_outputs()[0].type) == "tensor(double)"

    def test_decisiontree_regressor(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X, y)
        initial_types = [("input", FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model,
            initial_types=initial_types,
            final_types=[("output4", None)],
            target_opset=TARGET_OPSET,
        )
        assert model_onnx is not None
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        assert sess.get_outputs()[0].name == "output4"

    def test_kmeans(self):
        model = KMeans()
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X, y)
        initial_types = [("input", FloatTensorType((None, X.shape[1])))]
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model,
                initial_types=initial_types,
                final_types=[("output4", None)],
                target_opset=TARGET_OPSET,
            )
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model,
                initial_types=initial_types,
                final_types=[("dup1", None), ("dup1", None)],
                target_opset=TARGET_OPSET,
            )
        model_onnx = convert_sklearn(
            model,
            initial_types=initial_types,
            final_types=[("output4", None), ("output5", None)],
            target_opset=TARGET_OPSET,
        )
        assert model_onnx is not None
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        assert sess.get_outputs()[0].name == "output4"
        assert sess.get_outputs()[1].name == "output5"


class ScalerMetaEstimator(BaseEstimator, TransformerMixin):
    """A simple meta-estimator wrapping a StandardScaler for testing purposes."""

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        return self.scaler_.transform(X)


def scaler_meta_parser(scope, model, inputs, custom_parsers=None):
    """Custom parser using the public parse_sklearn_submodel API."""
    return parse_sklearn_submodel(
        scope, model.scaler_, inputs, custom_parsers=custom_parsers
    )


class TestParseSklearnSubmodel(unittest.TestCase):
    def test_parse_sklearn_submodel_importable(self):
        # Ensure the public API is importable from skl2onnx
        from skl2onnx import parse_sklearn_submodel as pss

        self.assertTrue(callable(pss))

    def test_parse_sklearn_submodel_in_custom_parser(self):
        X, _ = make_regression(n_features=4, n_samples=20, random_state=42)
        X = X.astype(np.float32)
        meta = ScalerMetaEstimator().fit(X)
        expected = meta.transform(X)

        initial_types = [("input", FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            meta,
            initial_types=initial_types,
            custom_parsers={ScalerMetaEstimator: scaler_meta_parser},
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": X})[0]
        assert_almost_equal(expected, got, decimal=5)


if __name__ == "__main__":
    unittest.main()
