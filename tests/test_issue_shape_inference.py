# SPDX-License-Identifier: Apache-2.0

import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from onnx.reference import ReferenceEvaluator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
)
import onnxruntime as rt


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
        X = X[:10]
        expected = model.predict(X).reshape((-1, 1))

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
                ("num_1", X.iloc[:, 0:1].astype(np.float32)),
                ("num_2", X.iloc[:, 1:2].astype(np.float32)),
                ("num_3", X.iloc[:, 2:3].astype(np.float32)),
                ("num_4", X.iloc[:, 3:4].astype(np.float32)),
                ("cat_1", X.iloc[:, 4:5].astype(np.int64)),
                ("cat_2", X.iloc[:, 5:6].astype(np.int64)),
            ]
        )

        # ReferenceEvaluator
        ref = ReferenceEvaluator(model_onnx, verbose=9)
        res = ref.run(None, feeds)
        self.assertEqual(1, len(res))
        self.assertEqual(expected.shape, res[0].shape)
        assert_almost_equal(expected, res[0])

        # onnxruntime
        sess = rt.InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        res = sess.run(None, feeds)
        self.assertEqual(1, len(res))
        self.assertEqual(expected.shape, res[0].shape)
        assert_almost_equal(expected, res[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
