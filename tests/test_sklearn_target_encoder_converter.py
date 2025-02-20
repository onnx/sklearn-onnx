# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's TargetEncoder converter."""

from skl2onnx import convert_sklearn, to_onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import __version__ as sklearn_version
import onnxruntime
import pandas as pd
from numpy.testing import assert_almost_equal
import unittest
import packaging.version as pv
import numpy as np
from onnxruntime import __version__ as ort_version
from skl2onnx.common.data_types import (
    Int64TensorType,
    StringTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET

try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    # Not available for scikit-learn < 1.3
    pass


ort_version = ".".join(ort_version.split(".")[:2])


def target_encoder_support():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    if pv.Version(vers) < pv.Version("1.3.0"):
        return False
    if pv.Version(onnxruntime.__version__) < pv.Version("0.3.0"):
        return False
    return pv.Version(vers) >= pv.Version("1.3.0")


def set_output_support():
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("1.2")


class TestSklearnTargetEncoderConverter(unittest.TestCase):
    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    def test_model_target_encoder(self):
        model = TargetEncoder()
        X = np.array(["str3", "str2", "str0", "str1", "str3"]).reshape(-1, 1)
        y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn target encoder",
            [("input", StringTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        if model_onnx.ir_version >= 7 and TARGET_OPSET < 12:
            raise AssertionError("Incompatbilities")
        dump_data_and_model(X, model, model_onnx, basename="SklearnTargetEncoder")

    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    def test_model_target_encoder_int(self):
        model = TargetEncoder()
        X = np.array([0, 0, 1, 0, 0, 1, 1], dtype=np.int64).reshape(-1, 1)
        y = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
        X_test = np.array([0, 1, 2, 1, 0], dtype=np.int64).reshape(-1, 1)

        model.fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn label encoder",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        if model_onnx.ir_version >= 7 and TARGET_OPSET < 12:
            raise AssertionError("Incompatbilities")
        dump_data_and_model(
            X_test, model, model_onnx, basename="SklearnTargetEncoderInt"
        )

    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    def test_target_encoder_twocats(self):
        data = [["cat2"], ["cat1"]]
        label = [0, 1]
        model = TargetEncoder(categories="auto")
        model.fit(data, label)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model, "Target encoder two string cats", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnTargetEncoderTwoStringCat"
        )

    @unittest.skipIf(
        not set_output_support(),
        reason="'ColumnTransformer' object has no attribute 'set_output'",
    )
    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    def test_target_encoder_pipeline_f32(self):
        from onnxruntime import InferenceSession

        data = pd.DataFrame({"cat": ["cat2", "cat1"] * 10, "num": [0, 1, 1, 0] * 5})
        data["num"] = data["num"].astype(np.float32)
        y = np.array([0, 1, 0, 1] * 5, dtype=np.float32)
        # target encoder uses cross-fitting and have cv=5 as default, which
        # caused some folds to have constant y.
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", TargetEncoder(cv=2), ["cat"]),
                ("num", "passthrough", ["num"]),
            ],
            sparse_threshold=1,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        model = make_pipeline(
            preprocessor, RandomForestRegressor(n_estimators=3, max_depth=2)
        )
        model.fit(data, y)
        expected = model.predict(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(
            None,
            {
                "cat": data["cat"].values.reshape((-1, 1)),
                "num": data["num"].values.reshape((-1, 1)),
            },
        )
        assert_almost_equal(expected, got[0].ravel())

    @unittest.skipIf(
        not set_output_support(),
        reason="'ColumnTransformer' object has no attribute 'set_output'",
    )
    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    def test_target_encoder_pipeline_string_int64(self):
        from onnxruntime import InferenceSession

        data = pd.DataFrame(
            {
                "C1": ["cat2", "cat1", "cat3"] * 10,
                "C2": [1, 0, 1] * 10,
                "num": [0, 1, 1] * 10,
            }
        )
        data["num"] = data["num"].astype(np.float32)
        data["C2"] = data["C2"].astype(np.int64)
        y = np.array([0, 1, 0, 1, 0, 1] * 5, dtype=np.int64)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", TargetEncoder(cv=2), ["C1", "C2"]),
                ("num", "passthrough", ["num"]),
            ],
            sparse_threshold=1,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        model = make_pipeline(
            preprocessor, RandomForestRegressor(n_estimators=3, max_depth=2)
        )
        model.fit(data, y)
        expected = model.predict(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(
            None,
            {
                "C1": data["C1"].values.reshape((-1, 1)),
                "C2": data["C2"].values.reshape((-1, 1)),
                "num": data["num"].values.reshape((-1, 1)),
            },
        )
        assert_almost_equal(expected, got[0].ravel())

    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_target_encoder_mixed_string_int_pandas(self):
        col1 = "col1"
        col2 = "col2"
        col3 = "col3"
        data_pd = pd.DataFrame(
            {
                col1: np.array(["c0.4", "c1.4", "c0.2", "c0.2", "c0.2", "c0.2"]),
                col2: np.array(["c0.2", "c1.2", "c2.2", "c2.2", "c2.2", "c2.2"]),
                col3: np.array([3, 0, 1, 1, 1, 1]),
            }
        )
        data_label = np.array([0, 0, 1, 1, 1, 0])
        test_pd = pd.DataFrame(
            {
                col1: np.array(["c0.2"]),
                col2: np.array(["c2.2"]),
                col3: np.array([1]),
            }
        )
        model = TargetEncoder()
        model.fit(data_pd, data_label)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "ordinal encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            test_pd,
            model,
            model_onnx,
            basename="SklearnTargetEncoderMixedStringIntPandas",
        )

    @unittest.skipIf(
        not target_encoder_support(),
        reason="TargetEncoder was not available before 1.3",
    )
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_target_encoder_multiclass_assertion(self):
        model = TargetEncoder()
        X = np.array([0, 0, 1, 0, 0, 1, 1], dtype=np.int64).reshape(-1, 1)
        y = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int64)

        with self.assertRaises(ValueError):
            # scikit-learn won't allow multiclass on 1.3.2
            model.fit(X, y)
            with self.assertRaises(NotImplementedError):
                # after that, we must ensure that the output is binary or continuous
                convert_sklearn(
                    model,
                    "scikit-learn target encoder",
                    [("input", Int64TensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
