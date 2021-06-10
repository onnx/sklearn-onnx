# SPDX-License-Identifier: Apache-2.0


import unittest
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from skl2onnx.common.data_types import onnx_built_with_ml
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType)
from skl2onnx import convert_sklearn
from onnxruntime import InferenceSession
from test_utils import TARGET_OPSET


class TestParsingOptions(unittest.TestCase):

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_pipeline(self):
        model = Pipeline(
            [('sc1', StandardScaler()), ('sc2', StandardScaler())])
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types,
                                     target_opset=TARGET_OPSET)
        assert model_onnx is not None
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            final_types=[('output', None)],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        assert sess.get_outputs()[0].name == 'output'
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            final_types=[('output4', None)],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        assert sess.get_outputs()[0].name == 'output4'
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            final_types=[('output4', DoubleTensorType())],
            target_opset=TARGET_OPSET)
        try:
            sess = InferenceSession(model_onnx.SerializeToString())
        except RuntimeError as e:
            if "Cast(9)" in str(e):
                return
            raise e
        assert sess.get_outputs()[0].name == 'output4'
        assert str(sess.get_outputs()[0].type) == "tensor(double)"

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decisiontree_regressor(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types,
                                     final_types=[('output4', None)],
                                     target_opset=TARGET_OPSET)
        assert model_onnx is not None
        sess = InferenceSession(model_onnx.SerializeToString())
        assert sess.get_outputs()[0].name == 'output4'

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_kmeans(self):
        model = KMeans()
        X, y = make_regression(n_features=4, random_state=42)
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        with self.assertRaises(RuntimeError):
            convert_sklearn(model, initial_types=initial_types,
                            final_types=[('output4', None)],
                            target_opset=TARGET_OPSET)
        with self.assertRaises(RuntimeError):
            convert_sklearn(model, initial_types=initial_types,
                            final_types=[('dup1', None), ('dup1', None)],
                            target_opset=TARGET_OPSET)
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            final_types=[('output4', None), ('output5', None)],
            target_opset=TARGET_OPSET)
        assert model_onnx is not None
        sess = InferenceSession(model_onnx.SerializeToString())
        assert sess.get_outputs()[0].name == 'output4'
        assert sess.get_outputs()[1].name == 'output5'


if __name__ == "__main__":
    unittest.main()
