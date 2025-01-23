# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's OneHotEncoder converter."""

import unittest
import sys
import packaging.version as pv
import numpy
from numpy.testing import assert_almost_equal
from onnx.defs import onnx_opset_version
import pandas

try:
    from onnx.reference import ReferenceEvaluator
except ImportError:
    ReferenceEvaluator = None
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    Int32TensorType,
    Int64TensorType,
    StringTensorType,
    FloatTensorType,
)
from skl2onnx.algebra.type_helper import guess_initial_types

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from test_utils import dump_data_and_model, TARGET_OPSET


def one_hot_encoder_supports_string():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("0.20.0")


def one_hot_encoder_supports_drop():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("0.21.0")


def skl12():
    # pv.Version does not work with development versions
    vers = ".".join(sklearn_version.split(".")[:2])
    return pv.Version(vers) >= pv.Version("1.2")


class TestSklearnOneHotEncoderConverter(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20",
    )
    def test_model_one_hot_encoder(self):
        model = OneHotEncoder(categories="auto")
        data = numpy.array(
            [[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=numpy.int64
        )
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderInt64-SkipDim1"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20",
    )
    def test_model_one_hot_encoder_int32(self):
        model = OneHotEncoder(categories="auto")
        data = numpy.array(
            [[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=numpy.int32
        )
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int32TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        str_model_onnx = str(model_onnx)
        assert "int64_data" in str_model_onnx
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderInt32-SkipDim1"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder did not have categories_ before 0.20",
    )
    @ignore_warnings(category=FutureWarning)
    @unittest.skipIf(not skl12(), reason="sparse_output")
    def test_model_one_hot_encoder_int32_scaler(self):
        model = make_pipeline(
            OneHotEncoder(categories="auto", sparse_output=False), RobustScaler()
        )
        data = numpy.array(
            [[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=numpy.int32
        )
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int32TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        str_model_onnx = str(model_onnx)
        assert "int64_data" in str_model_onnx
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderInt32Scaler-SkipDim1"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21",
    )
    def test_one_hot_encoder_mixed_string_int_drop(self):
        data = [
            ["c0.4", "c0.2", 3],
            ["c1.4", "c1.2", 0],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
        ]
        test = [["c0.2", "c2.2", 1]]
        model = OneHotEncoder(categories="auto", drop=["c0.4", "c0.2", 3])
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnOneHotEncoderMixedStringIntDrop",
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19",
    )
    def test_one_hot_encoder_onecat(self):
        data = [["cat"], ["cat"]]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model, "one-hot encoder one string cat", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderOneStringCat"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support strings in 0.19",
    )
    def test_one_hot_encoder_twocats(self):
        data = [["cat2"], ["cat1"]]
        model = OneHotEncoder(categories="auto")
        model.fit(data)
        inputs = [("input1", StringTensorType([None, 1]))]
        model_onnx = convert_sklearn(
            model, "one-hot encoder two string cats", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderTwoStringCat"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21",
    )
    def test_one_hot_encoder_string_drop_first(self):
        data = [["Male", "First"], ["Female", "First"], ["Female", "Second"]]
        test_data = [["Male", "Second"]]
        model = OneHotEncoder(drop="first", categories="auto")
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 1])),
            ("input2", StringTensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test_data, model, model_onnx, basename="SklearnOneHotEncoderStringDropFirst"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support this in 0.19",
    )
    @ignore_warnings(category=FutureWarning)
    @unittest.skipIf(not skl12(), reason="sparse_output")
    def test_model_one_hot_encoder_list_sparse(self):
        model = OneHotEncoder(
            categories=[[0, 1, 4, 5], [1, 2, 3, 5], [0, 3, 4, 6]], sparse_output=True
        )
        data = numpy.array(
            [[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=numpy.int64
        )
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input1", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderCatSparse-SkipDim1"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_string(),
        reason="OneHotEncoder does not support this in 0.19",
    )
    @ignore_warnings(category=FutureWarning)
    @unittest.skipIf(not skl12(), reason="sparse_output")
    def test_model_one_hot_encoder_list_dense(self):
        model = OneHotEncoder(
            categories=[[0, 1, 4, 5], [1, 2, 3, 5], [0, 3, 4, 6]], sparse_output=False
        )
        data = numpy.array(
            [[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=numpy.int64
        )
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "scikit-learn one-hot encoder",
            [("input", Int64TensorType([None, 3]))],
            target_opset=TARGET_OPSET,
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderCatDense-SkipDim1"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21",
    )
    def test_one_hot_encoder_int_drop(self):
        data = [
            [1, 2, 3],
            [4, 1, 0],
            [0, 2, 1],
            [2, 2, 1],
            [0, 4, 0],
            [0, 3, 3],
        ]
        test = numpy.array([[2, 2, 1], [4, 2, 1]], dtype=numpy.int64)
        model = OneHotEncoder(categories="auto", drop=[0, 1, 3], dtype=numpy.float32)
        model.fit(data)
        inputs = [
            ("input1", Int64TensorType([None, 3])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test, model, model_onnx, basename="SklearnOneHotEncoderIntDrop"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21",
    )
    def test_one_hot_encoder_int_drop_first(self):
        data = [
            [1, 2, 3],
            [4, 1, 0],
            [0, 2, 1],
            [2, 2, 1],
            [0, 4, 0],
            [0, 3, 3],
        ]
        test = numpy.array([[2, 2, 1], [1, 3, 3]], dtype=numpy.int64)
        model = OneHotEncoder(categories="auto", drop="first", dtype=numpy.int64)
        model.fit(data)
        inputs = [
            ("input1", Int64TensorType([None, 3])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test, model, model_onnx, basename="SklearnOneHotEncoderIntDropFirst"
        )

    @unittest.skipIf(
        pv.Version(ort_version) <= pv.Version("0.4.0"), reason="issues with shapes"
    )
    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21",
    )
    def test_one_hot_encoder_string_drop_first_2(self):
        data = [["Male", "First"], ["Female", "First"], ["Female", "Second"]]
        model = OneHotEncoder(drop="first")
        model.fit(data)
        inputs = [
            ("input", StringTensorType([None, 2])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="SklearnOneHotEncoderStringDropFirst2"
        )

    def _shape_inference(self, engine):
        cat_columns_openings = ["cat_1", "cat_2"]
        num_columns_openings = ["num_1", "num_2", "num_3", "num_4"]

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
        num_data = numpy.random.rand(100, 4)
        cat_data = numpy.random.randint(11, size=(100, 2))
        df = pandas.DataFrame(
            numpy.hstack((num_data, cat_data)),
            columns=["num_1", "num_2", "num_3", "num_4", "cat_1", "cat_2"],
        )
        df[num_columns_openings] = df[num_columns_openings].astype(float)
        df[cat_columns_openings] = df[cat_columns_openings].astype(int)
        df["target"] = numpy.random.rand(100)
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

        model_onnx = convert_sklearn(
            model, initial_types=initial_type, target_opset=TARGET_OPSET
        )
        if TARGET_OPSET < 19:
            model_onnx.ir_version = 8

        feeds = dict(
            [
                ("num_1", X.iloc[:, 0:1].values.astype(numpy.float32)),
                ("num_2", X.iloc[:, 1:2].values.astype(numpy.float32)),
                ("num_3", X.iloc[:, 2:3].values.astype(numpy.float32)),
                ("num_4", X.iloc[:, 3:4].values.astype(numpy.float32)),
                ("cat_1", X.iloc[:, 4:5].values.astype(numpy.int64)),
                ("cat_2", X.iloc[:, 5:6].values.astype(numpy.int64)),
            ]
        )

        # onnxruntime
        if engine == "onnxruntime":
            ref = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        else:
            ref = ReferenceEvaluator(model_onnx)

        res = ref.run(None, feeds)
        self.assertEqual(1, len(res))
        self.assertEqual(expected.shape, res[0].shape)
        assert_almost_equal(expected, res[0])

    @unittest.skipIf(
        onnx_opset_version() < 19, reason="missing ops in reference implementation"
    )
    @ignore_warnings(category=RuntimeWarning)
    @unittest.skipIf(sys.platform == "darwin", "interesting discrepancy")
    def test_shape_inference_onnx(self):
        self._shape_inference("onnx")

    @unittest.skipIf(
        onnx_opset_version() < 19, reason="missing ops in reference implementation"
    )
    @ignore_warnings(category=RuntimeWarning)
    @unittest.skipIf(sys.platform == "darwin", "interesting discrepancy")
    def test_shape_inference_onnxruntime(self):
        self._shape_inference("onnxruntime")

    @unittest.skipIf(not skl12(), reason="sparse output not available")
    def test_min_frequency(self):
        data = pandas.DataFrame(
            [
                dict(CAT1="aa", CAT2="ba", num1=0.5, num2=0.6, y=0),
                dict(CAT1="ab", CAT2="bb", num1=0.4, num2=0.8, y=1),
                dict(CAT1="ac", CAT2="bb", num1=0.4, num2=0.8, y=1),
                dict(CAT1="ab", CAT2="bc", num1=0.5, num2=0.56, y=0),
                dict(CAT1="ab", CAT2="bd", num1=0.55, num2=0.56, y=1),
                dict(CAT1="ab", CAT2="bd", num1=0.35, num2=0.86, y=0),
                dict(CAT1="ab", CAT2="bd", num1=0.5, num2=0.68, y=1),
            ]
        )
        cat_cols = ["CAT1", "CAT2"]
        train_data = data.drop("y", axis=1)
        for c in train_data.columns:
            if c not in cat_cols:
                train_data[c] = train_data[c].astype(numpy.float32)

        pipe = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                Pipeline(
                                    [
                                        (
                                            "onehot",
                                            OneHotEncoder(
                                                min_frequency=2,
                                                sparse_output=False,
                                                handle_unknown="ignore",
                                            ),
                                        )
                                    ]
                                ),
                                cat_cols,
                            )
                        ],
                        remainder="passthrough",
                    ),
                ),
            ]
        )
        pipe.fit(train_data, data["y"])

        init = guess_initial_types(train_data, None)
        self.assertEqual([i[0] for i in init], "CAT1 CAT2 num1 num2".split())
        for t in init:
            self.assertEqual(t[1].shape, [None, 1])
        onx2 = to_onnx(pipe, initial_types=init)
        with open("kkkk.onnx", "wb") as f:
            f.write(onx2.SerializeToString())
        sess2 = InferenceSession(
            onx2.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        inputs = {c: train_data[c].values.reshape((-1, 1)) for c in train_data.columns}
        got2 = sess2.run(None, inputs)

        expected = pipe.transform(train_data)
        assert_almost_equal(expected, got2[0])

    @unittest.skipIf(
        not one_hot_encoder_supports_drop(),
        reason="OneHotEncoder does not support drop in scikit versions < 0.21",
    )
    def test_one_hot_encoder_drop_if_binary(self):
        data = [
            ["c0.4", "c0.2", 0],
            ["c1.4", "c1.2", 0],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
            ["c0.2", "c2.2", 1],
        ]
        test = [["c0.2", "c2.2", 1]]
        model = OneHotEncoder(categories="auto", drop="if_binary")
        model.fit(data)
        inputs = [
            ("input1", StringTensorType([None, 2])),
            ("input2", Int64TensorType([None, 1])),
        ]
        model_onnx = convert_sklearn(
            model, "one-hot encoder", inputs, target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            test,
            model,
            model_onnx,
            verbose=False,
            basename="SklearnOneHotEncoderMixedStringIntDrop",
        )


if __name__ == "__main__":
    import logging

    for name in ["skl2onnx"]:
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
    # TestSklearnOneHotEncoderConverter().test_min_frequency()
    # TestSklearnOneHotEncoderConverter().test_one_hot_encoder_drop_if_binary()
    unittest.main(verbosity=2)
