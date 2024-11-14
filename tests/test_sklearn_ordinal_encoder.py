# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's OrdinalEncoder converter."""

import unittest
from numpy.testing import assert_almost_equal
import packaging.version as pv
import numpy as np
import pandas as pd
import onnxruntime
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    pass
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    Int64TensorType,
    StringTensorType,
)
from test_utils import dump_data_and_model, TARGET_OPSET


def ordinal_encoder_support():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    if pv.Version(vers) < pv.Version("0.20.0"):
        return False
    if pv.Version(onnxruntime.__version__) < pv.Version("0.3.0"):
        return False
    return pv.Version(vers) >= pv.Version("0.20.0")


def set_output_support():
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("1.2")


class TestSklearnOrdinalEncoderConverter(unittest.TestCase):
    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder(self):
        model = OrdinalEncoder(dtype=np.int64)
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn ordinal encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOrdinalEncoderInt64-SkipDim1"
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_ordinal_encoder_mixed_string_int_drop(self):
        data = [
            ["c0.4", "c0.2", 3],
            ["c1.4", "c1.2", 0],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
        ]
        test = [["c0.2", "c2.2", 1]]
        model = OrdinalEncoder(categories="auto")
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "ordinal encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test, model, model_onnx, basename="SklearnOrdinalEncoderMixedStringIntDrop"
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_ordinal_encoder_mixed_string_int_pandas(self):
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
        test_pd = pd.DataFrame(
            {
                col1: np.array(["c0.2"]),
                col2: np.array(["c2.2"]),
                col3: np.array([1]),
            }
        )
        model = OrdinalEncoder(categories="auto")
        model.fit(data_pd)
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
            basename="SklearnOrdinalEncoderMixedStringIntPandas",
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_ordinal_encoder_onecat(self):
        data = [["cat"], ["cat"]]
        model = OrdinalEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model, "ordinal encoder one string cat", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOrdinalEncoderOneStringCat"
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_ordinal_encoder_twocats(self):
        data = [["cat2"], ["cat1"]]
        model = OrdinalEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model, "ordinal encoder two string cats", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOrdinalEncoderTwoStringCat"
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder_cat_list(self):
        model = OrdinalEncoder(categories=[[0, 1, 4, 5], [1, 2, 3, 5], [0, 3, 4, 6]])
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.int64)
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn ordinal encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOrdinalEncoderCatList"
        )

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder_unknown_value(self):
        from onnxruntime import InferenceSession

        model = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=42)
        data = np.array([["a"], ["b"], ["c"], ["d"]], dtype=np.object_)
        data_with_missing_value = np.array(
            [["a"], ["b"], ["c"], ["d"], [np.nan], ["e"], [None]], dtype=np.object_
        )

        model.fit(data)
        # 'np.nan','e' and 'None' become 42.
        expected = model.transform(data_with_missing_value)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn ordinal encoder",
            [("input", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOrdinalEncoderUnknownValue"
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(
            None,
            {
                "input": data_with_missing_value,
            },
        )

        assert_almost_equal(expected.reshape(-1), got[0].reshape(-1))

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder_encoded_missing_value(self):
        from onnxruntime import InferenceSession

        model = OrdinalEncoder(encoded_missing_value=42)
        data = np.array([["a"], ["b"], [np.nan], ["c"], ["d"]], dtype=np.object_)

        # 'np.nan' becomes 42
        expected = model.fit_transform(data)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn ordinal encoder",
            [("input", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOrdinalEncoderEncodedMissingValue"
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(
            None,
            {
                "input": data,
            },
        )

        assert_almost_equal(expected.reshape(-1), got[0].reshape(-1))

    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_model_ordinal_encoder_encoded_missing_value_no_nan(self):
        from onnxruntime import InferenceSession

        model = OrdinalEncoder(encoded_missing_value=42)
        data = np.array([["a"], ["b"], ["c"], ["d"]], dtype=np.object_)

        expected = model.fit_transform(data)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn ordinal encoder",
            [("input", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnOrdinalEncoderEncodedMissingValueNoNan",
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(
            None,
            {
                "input": data,
            },
        )

        assert_almost_equal(expected.reshape(-1), got[0].reshape(-1))

    @unittest.skipIf(
        not set_output_support(),
        reason="'ColumnTransformer' object has no attribute 'set_output'",
    )
    @unittest.skipIf(
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_ordinal_encoder_pipeline_int64(self):
        from onnxruntime import InferenceSession

        data = pd.DataFrame({"cat": ["cat2", "cat1"], "num": [0, 1]})
        data["num"] = data["num"].astype(np.float32)
        y = np.array([0, 1], dtype=np.float32)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OrdinalEncoder(dtype=np.int64), ["cat"]),
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
        model_onnx = to_onnx(model, data[:1], target_opset=TARGET_OPSET)
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
        not ordinal_encoder_support(),
        reason="OrdinalEncoder was not available before 0.20",
    )
    def test_ordinal_encoder_pipeline_string_int64(self):
        from onnxruntime import InferenceSession

        data = pd.DataFrame(
            {"C1": ["cat2", "cat1", "cat3"], "C2": [1, 0, 1], "num": [0, 1, 1]}
        )
        data["num"] = data["num"].astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.float32)
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OrdinalEncoder(dtype=np.int64), ["C1", "C2"]),
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
        model_onnx = to_onnx(model, data[:1], target_opset=TARGET_OPSET)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
