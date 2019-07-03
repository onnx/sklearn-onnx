"""
Tests scikit-imputer converter.
"""
import unittest
import numpy as np
import pandas
from sklearn.datasets import load_digits, load_iris
from sklearn.pipeline import Pipeline

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model


class TestSklearnFunctionTransformerConverter(unittest.TestCase):
    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer introduced in 0.20",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_function_transformer(self):
        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == "int64":
                    t = Int64TensorType([1, 1])
                elif v == "float64":
                    t = FloatTensorType([1, 1])
                else:
                    t = StringTensorType([1, 1])
                inputs.append((k, t))
            return inputs

        data = load_digits()
        X = data.data[:, :2]
        y = data.target
        data = pandas.DataFrame(X, columns=["X1", "X2"], dtype=np.int64)

        pipe = Pipeline(steps=[
            (
                "select",
                ColumnTransformer([("id", FunctionTransformer(),
                                    ["X1", "X2"])]),
            ),
            ("logreg", LogisticRegression()),
        ])
        pipe.fit(data[["X1", "X2"]], y)

        inputs = convert_dataframe_schema(data)
        model_onnx = convert_sklearn(pipe, "scikit-learn function_transformer",
                                     inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data[:5],
            pipe,
            model_onnx,
            basename="SklearnFunctionTransformer-DF",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer introduced in 0.20",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_passthrough(self):
        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == "int64":
                    t = Int64TensorType([1, 1])
                elif v == "float64":
                    t = FloatTensorType([1, 1])
                else:
                    t = StringTensorType([1, 1])
                inputs.append((k, t))
            return inputs

        data = load_iris()
        X = data.data[:, :2]
        y = data.target
        data = pandas.DataFrame(X, columns=["X1", "X2"])

        pipe = Pipeline(steps=[
            ("select",
                ColumnTransformer([("id", FunctionTransformer(), ["X1"]),
                                   ("id2", "passthrough", ["X2"])])),
            ("logreg", LogisticRegression()),
        ])
        pipe.fit(data[["X1", "X2"]], y)

        inputs = convert_dataframe_schema(data)
        model_onnx = convert_sklearn(pipe, "scikit-learn function_transformer",
                                     inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data[:5],
            pipe,
            model_onnx,
            basename="SklearnFunctionTransformer-DF",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer introduced in 0.20",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_remainder_passthrough(self):
        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == "int64":
                    t = Int64TensorType([1, 1])
                elif v == "float64":
                    t = FloatTensorType([1, 1])
                else:
                    t = StringTensorType([1, 1])
                inputs.append((k, t))
            return inputs

        data = load_iris()
        X = data.data[:, :2]
        y = data.target
        data = pandas.DataFrame(X, columns=["X1", "X2"])

        pipe = Pipeline(steps=[
            ("select",
                ColumnTransformer([("id", FunctionTransformer(), ["X1"])],
                                  remainder="passthrough")),
            ("logreg", LogisticRegression()),
        ])
        pipe.fit(data[["X1", "X2"]], y)

        inputs = convert_dataframe_schema(data)
        model_onnx = convert_sklearn(pipe, "scikit-learn function_transformer",
                                     inputs)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data[:5],
            pipe,
            model_onnx,
            basename="SklearnFunctionTransformer-DF",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
