# SPDX-License-Identifier: Apache-2.0

import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


class TestIssueShapeInference(unittest.TestCase):
    
    def test_shape_inference(self):

        cat_columns_openings = ["cat_1", "cat_2"]
        num_columns_openings = [
            "num_1",
            "num_2",
            "num_3",
            "num_4",
        ]

        regression_aperturas = LinearRegression()

        numeric_transformer = SimpleImputer(strategy="median")
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )


        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_columns_openings),
                ("cat", categorical_transformer, cat_columns_openings),
            ]
        )

        model = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", regression_aperturas)]
        )

        # Create sample df
        num_data = np.random.rand(100, 4)
        cat_data = np.random.randint(11, size=(100, 2))
        df = pd.DataFrame(
            np.hstack((num_data, cat_data)),
            columns=["num_1", "num_2", "num_3", "num_4", "cat_1", "cat_2"],
        )
        df[num_columns_openings] = df[num_columns_openings].astype(float)
        df[cat_columns_openings] = df[cat_columns_openings].astype(int)
        df["target"] = np.random.rand(100)
        df["target"] = df["target"].astype(float)
        X = df.drop("target", axis=1)
        y = df["target"]
        model.fit(X, y)

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import (
            FloatTensorType,
            StringTensorType,
            Int64TensorType,
        )
        import onnxruntime as rt
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
        from onnx.reference import ReferenceEvaluator
        from onnx.shape_inference import infer_shapes
        from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs


        initial_type = [
            ("num_1", FloatTensorType([None, 1])),
            ("num_2", FloatTensorType([None, 1])),
            ("num_3", FloatTensorType([None, 1])),
            ("num_4", FloatTensorType([None, 1])),
            ("cat_1", Int64TensorType([None, 1])),
            ("cat_2", Int64TensorType([None, 1])),
        ]

        model_onnx = convert_sklearn(model, initial_types=initial_type)

        feeds = dict(
            [
                ("num_1", np.array([[1], [2]], dtype=np.float32)),
                ("num_2", np.array([[1], [2]], dtype=np.float32)),
                ("num_3", np.array([[1], [2]], dtype=np.float32)),
                ("num_4", np.array([[1], [2]], dtype=np.float32)),
                ("cat_1", np.array([[1], [2]], dtype=np.int64)),
                ("cat_2", np.array([[1], [2]], dtype=np.int64)),
            ]
        )

        ref = ReferenceEvaluator(model_onnx, verbose=9)
        expected = ref.run(None, feeds)

        sess = rt.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, feeds)
        self.assertEqual(len(expected), len(res))
        for a, b in zip(expected, res):
            self.assertEqual(a.shape, b.shape)
            assert_almost_equal(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
