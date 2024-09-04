# SPDX-License-Identifier: Apache-2.0
"""
Tests scikit-learn's feature selection converters
"""

import unittest
from typing import Optional
import packaging.version as pv
import numpy as np
from sklearn.utils._testing import assert_almost_equal
from pandas import DataFrame
from onnx import TensorProto, __version__ as onnx_version
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model

try:
    from onnx.reference import ReferenceEvaluator
    from onnx.reference.op_run import OpRun
except ImportError:
    ReferenceEvaluator = None
from onnxruntime import __version__ as ort_version, SessionOptions
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import to_onnx
from skl2onnx.common.data_types import (
    StringTensorType,
    Int64TensorType,
    FloatTensorType,
    DoubleTensorType,
)
from test_utils import (
    TARGET_OPSET,
    TARGET_IR,
    InferenceSessionEx as InferenceSession,
)


class TestSklearnFeatureHasher(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.12.0"), reason="no murmurhash3 in ort"
    )
    def test_ort_murmurhash3_int(self):
        X = make_tensor_value_info("X", TensorProto.UINT32, [None])
        Y = make_tensor_value_info("Y", TensorProto.UINT32, [None])
        node = make_node(
            "MurmurHash3", ["X"], ["Y"], domain="com.microsoft", positive=1, seed=0
        )
        graph = make_graph([node], "hash", [X], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("", TARGET_OPSET),
                make_opsetid("com.microsoft", 1),
            ],
            ir_version=TARGET_IR,
        )
        check_model(onnx_model)
        sess = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = {"X": np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)}
        got = sess.run(None, feeds)
        self.assertEqual(got[0].shape, feeds["X"].shape)
        self.assertEqual(got[0].dtype, feeds["X"].dtype)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.12.0"), reason="no murmurhash3 in ort"
    )
    def test_ort_murmurhash3_string(self):
        X = make_tensor_value_info("X", TensorProto.STRING, [None])
        Y = make_tensor_value_info("Y", TensorProto.INT32, [None])
        node = make_node(
            "MurmurHash3", ["X"], ["Y"], domain="com.microsoft", positive=0, seed=0
        )
        graph = make_graph([node], "hash", [X], [Y])
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("", TARGET_OPSET),
                make_opsetid("com.microsoft", 1),
            ],
            ir_version=TARGET_IR,
        )
        check_model(onnx_model)
        sess = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        input_strings = ["z0", "o11", "d222", "q4444", "t333", "c5555"]
        feeds = {"X": np.array(input_strings)}
        got = sess.run(None, feeds)

        n_features = 4

        res = got[0]

        ind = res == np.int32(-2147483648)
        indices = res.copy()
        indices[ind] = (2147483647 - (n_features - 1)) % n_features
        indices[~ind] = np.abs(indices[~ind]) % n_features

        final = np.where(res >= 0, 1, 4294967295).astype(np.uint32)
        mat = np.zeros((res.shape[0], n_features), dtype=np.uint32)
        for i in range(final.shape[0]):
            mat[i, indices[i]] = final[i]

        skl = FeatureHasher(n_features, input_type="string", dtype=np.uint32)
        expected = skl.transform(feeds["X"].reshape((-1, 1)))
        dense = expected.todense()
        for i, (a, b) in enumerate(zip(dense.tolist(), mat.tolist())):
            if a != b:
                raise AssertionError(f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher(self):
        n_features = 5
        input_strings = ["z0", "o11", "d222", "q4444", "t333", "c5555"]
        data = np.array(input_strings).reshape((-1, 1))
        for alternate_sign, dtype in [
            (True, np.float32),
            (True, np.float64),
            (True, np.int64),
            (False, np.float32),
        ]:
            if dtype == np.float32:
                final_type = FloatTensorType
            elif dtype == np.float64:
                final_type = DoubleTensorType
            elif dtype in (np.int32, np.uint32, np.int64):
                final_type = Int64TensorType
            else:
                final_type = None
            with self.subTest(alternate_sign=alternate_sign, dtype=dtype):
                model = FeatureHasher(
                    n_features=n_features,
                    alternate_sign=alternate_sign,
                    dtype=dtype,
                    input_type="string",
                )
                model.fit(data)
                expected = model.transform(data).todense()

                model_onnx = to_onnx(
                    model,
                    initial_types=[("X", StringTensorType([None, 1]))],
                    target_opset=TARGET_OPSET,
                    final_types=[("Y", final_type([None, 1]))],
                )
                sess = InferenceSession(
                    model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, {"X": data})
                self.assertEqual(expected.shape, got[0].shape)
                self.assertEqual(expected.dtype, got[0].dtype)
                for i, (a, b) in enumerate(zip(expected.tolist(), got[0].tolist())):
                    if a != b:
                        raise AssertionError(f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_two_columns(self):
        n_features = 5
        input_strings = ["z0", "o11", "d222", "q4444", "t333", "c5555"]
        data = np.array(input_strings).reshape((-1, 2))

        model = FeatureHasher(
            n_features=n_features,
            alternate_sign=True,
            dtype=np.float32,
            input_type="string",
        )
        model.fit(data)
        expected = model.transform(data).todense()

        model_onnx = to_onnx(
            model,
            initial_types=[("X", StringTensorType([None, data.shape[1]]))],
            target_opset=TARGET_OPSET,
            final_types=[("Y", FloatTensorType([None, n_features]))],
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": data})
        self.assertEqual(expected.shape, got[0].shape)
        self.assertEqual(expected.dtype, got[0].dtype)
        for i, (a, b) in enumerate(zip(expected.tolist(), got[0].tolist())):
            if a != b:
                raise AssertionError(f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_dataframe(self):
        n_features = 5
        input_strings = ["z0", "o11", "d222", "q4444", "t333", "c5555"]
        data = np.array(input_strings).reshape((-1, 2))
        data = DataFrame(data)
        data.columns = ["c1", "c2"]
        data_nx = data.values

        # The code of the feature hasher produces this intermediate
        # representation very different if the input is a dataframe.
        # The unit test is valid if both expressions produces the same results
        # otherwise, numpy arrays must be used.
        df = [[(f, 1) for f in x] for x in data]
        ar = [[(f, 1) for f in x] for x in data.values]
        if df != ar:
            return

        model = FeatureHasher(
            n_features=n_features,
            alternate_sign=True,
            dtype=np.float32,
            input_type="string",
        )
        model.fit(data)
        expected = model.transform(data).todense()

        model_onnx = to_onnx(
            model,
            initial_types=[("X", StringTensorType([None, data.shape[0]]))],
            target_opset=TARGET_OPSET,
            final_types=[("Y", FloatTensorType([None, n_features]))],
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": data_nx})
        self.assertEqual(expected.shape, got[0].shape)
        self.assertEqual(expected.dtype, got[0].dtype)
        for i, (a, b) in enumerate(zip(expected.tolist(), got[0].tolist())):
            if a != b:
                raise AssertionError(f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_two_columns_unicode(self):
        n_features = 5
        input_strings = ["z0", "o11", "d222", "고리", "é", "ô"]
        data = np.array(input_strings).reshape((-1, 2))

        model = FeatureHasher(
            n_features=n_features,
            alternate_sign=True,
            dtype=np.float32,
            input_type="string",
        )
        model.fit(data)
        expected = model.transform(data).todense()

        model_onnx = to_onnx(
            model,
            initial_types=[("X", StringTensorType([None, data.shape[1]]))],
            target_opset=TARGET_OPSET,
            final_types=[("Y", FloatTensorType([None, n_features]))],
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"X": data})
        self.assertEqual(expected.shape, got[0].shape)
        self.assertEqual(expected.dtype, got[0].dtype)
        for i, (a, b) in enumerate(zip(expected.tolist(), got[0].tolist())):
            if a != b:
                raise AssertionError(f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_pipeline(self):
        data = {
            "Education": ["a", "b", "d", "abd"],
            "Label": [1, 1, 0, 0],
        }
        df = DataFrame(data)

        cat_features = ["Education"]
        X_train = df[cat_features]

        X_train["cat_features"] = df[cat_features].values.tolist()
        X_train = X_train.drop(cat_features, axis=1)
        y_train = df["Label"]

        preprocessing_pipeline = ColumnTransformer(
            [
                (
                    "cat_preprocessor",
                    FeatureHasher(
                        n_features=16,
                        input_type="string",
                        alternate_sign=False,
                        dtype=np.float32,
                    ),
                    "cat_features",
                )
            ],
            sparse_threshold=0.0,
        )

        complete_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessing_pipeline),
                ("classifier", DecisionTreeClassifier(max_depth=2)),
            ],
        )
        complete_pipeline.fit(X_train, y_train)

        # first check
        model = FeatureHasher(
            n_features=16,
            input_type="string",
            alternate_sign=False,
            dtype=np.float32,
        )
        X_train_ort1 = X_train.values.reshape((-1, 1))
        with self.assertRaises(TypeError):
            np.asarray(model.transform(X_train_ort1).todense())
        input_strings = ["a", "b", "d", "abd"]
        X_train_ort2 = np.array(input_strings, dtype=object).reshape((-1, 1))
        model.fit(X_train_ort2)
        # type(X_train_ort2[0, 0]) == str != list == type(X_train_ort2[0, 0])
        expected2 = np.asarray(model.transform(X_train_ort2).todense())
        model_onnx = to_onnx(
            model,
            initial_types=[("cat_features", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got2 = sess.run(None, dict(cat_features=X_train_ort2))
        assert_almost_equal(expected2, got2[0])
        got1 = sess.run(None, dict(cat_features=X_train_ort1))
        with self.assertRaises(AssertionError):
            assert_almost_equal(expected2, got1[0])

        # check hash
        X_train_ort = X_train.values
        expected = np.asarray(
            complete_pipeline.steps[0][-1]
            .transformers_[0][1]
            .transform(X_train.values.ravel())
            .todense()
        )
        model_onnx = to_onnx(
            complete_pipeline.steps[0][-1].transformers_[0][1],
            initial_types=[("cat_features", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, dict(cat_features=X_train_ort))
        with self.assertRaises(AssertionError):
            assert_almost_equal(expected, got[0])
        got = sess.run(None, dict(cat_features=X_train_ort2))
        assert_almost_equal(expected, got[0])

        # transform
        X_train_ort = X_train.values
        expected = complete_pipeline.steps[0][-1].transform(X_train)
        model_onnx = to_onnx(
            complete_pipeline.steps[0][-1],
            initial_types=[("cat_features", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, dict(cat_features=X_train_ort))
        with self.assertRaises(AssertionError):
            assert_almost_equal(expected, got[0].astype(np.float64))
        got = sess.run(None, dict(cat_features=X_train_ort2))
        assert_almost_equal(expected, got[0].astype(np.float64))

        # classifier
        expected = complete_pipeline.predict_proba(X_train)
        labels = complete_pipeline.predict(X_train)
        model_onnx = to_onnx(
            complete_pipeline,
            initial_types=[("cat_features", StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={"zipmap": False},
        )

        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        X_train_ort = X_train.values
        got = sess.run(None, dict(cat_features=X_train_ort))
        with self.assertRaises(AssertionError):
            assert_almost_equal(expected, got[1].astype(np.float64))
        got = sess.run(None, dict(cat_features=X_train_ort2))
        assert_almost_equal(labels, got[0])

    @unittest.skipIf(
        pv.Version(onnx_version) < pv.Version("1.11"), reason="onnx is too old"
    )
    def test_feature_hasher_pipeline_list(self):
        pipe_hash = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            (
                                "cat_features",
                                FeatureHasher(
                                    n_features=8,
                                    input_type="string",
                                    alternate_sign=False,
                                    dtype=np.float32,
                                ),
                                "cat_features",
                            ),
                        ],
                        sparse_threshold=0.0,
                    ),
                ),
            ],
        )

        df = DataFrame(
            {
                "Cat1": ["a", "b", "d", "abd", "e", "z", "ez"],
                "Cat2": ["A", "B", "D", "ABD", "e", "z", "ez"],
            }
        )

        cat_features = [c for c in df.columns if "Cat" in c]
        X_train = df[cat_features].copy()
        X_train["cat_features"] = df[cat_features].values.tolist()
        X_train = X_train.drop(cat_features, axis=1)
        pipe_hash.fit(X_train)
        expected = pipe_hash.transform(X_train)

        onx = to_onnx(
            pipe_hash,
            initial_types=[("cat_features", StringTensorType([None, 1]))],
            options={FeatureHasher: {"separator": "#"}},
            target_opset=TARGET_OPSET,
        )

        dfx = df.copy()
        dfx["cat_features"] = df[cat_features].agg("#".join, axis=1)
        feeds = dict(cat_features=dfx["cat_features"].values.reshape((-1, 1)))

        if ReferenceEvaluator is not None:

            class StringSplit(OpRun):
                op_domain = "ai.onnx.contrib"

                def _run(self, input, separator, skip_empty, **kwargs):
                    # kwargs should be null, bug in onnx?
                    delimiter = (
                        str(separator[0])
                        if len(separator.shape) > 0
                        else str(separator)
                    )
                    skip_empty = (
                        bool(skip_empty[0])
                        if len(skip_empty.shape)
                        else bool(skip_empty)
                    )
                    texts = []
                    indices = []
                    max_split = 0
                    for row, text in enumerate(input):
                        if not text:
                            continue
                        res = text.split(delimiter)
                        if skip_empty:
                            res = [t for t in res if t]
                        texts.extend(res)
                        max_split = max(max_split, len(res))
                        indices.extend((row, i) for i in range(len(res)))
                    return (
                        np.array(indices, dtype=np.int64),
                        np.array(texts),
                        np.array([len(input), max_split], dtype=np.int64),
                    )

            class MurmurHash3(OpRun):
                op_domain = "com.microsoft"

                @staticmethod
                def rotl(num, bits):
                    bit = num & (1 << (bits - 1))
                    num <<= 1
                    if bit:
                        num |= 1
                    num &= 2**bits - 1
                    return num

                @staticmethod
                def fmix(h: int):
                    h ^= h >> 16
                    h = np.uint32(
                        (int(h) * int(0x85EBCA6B)) % (int(np.iinfo(np.uint32).max) + 1)
                    )
                    h ^= h >> 13
                    h = np.uint32(
                        (int(h) * int(0xC2B2AE35)) % (int(np.iinfo(np.uint32).max) + 1)
                    )
                    h ^= h >> 16
                    return h

                @staticmethod
                def MurmurHash3_x86_32(data, seed):
                    le = len(data)
                    nblocks = le // 4
                    h1 = seed

                    c1 = 0xCC9E2D51
                    c2 = 0x1B873593

                    iblock = nblocks * 4

                    for i in range(-nblocks, 0):
                        k1 = np.uint32(data[iblock + i])
                        k1 *= c1
                        k1 = (k1, 15)
                        k1 *= c2
                        h1 ^= k1
                        h1 = MurmurHash3.rotl(h1, 13)
                        h1 = h1 * 5 + 0xE6546B64

                    k1 = 0

                    if le & 3 >= 3:
                        k1 ^= np.uint32(data[iblock + 2]) << 16
                    if le & 3 >= 2:
                        k1 ^= np.uint32(data[iblock + 1]) << 8
                    if le & 3 >= 1:
                        k1 ^= np.uint32(data[iblock])
                        k1 *= c1
                        k1 = MurmurHash3.rotl(k1, 15)
                        k1 *= c2
                        h1 ^= k1

                    h1 ^= le

                    h1 = MurmurHash3.fmix(h1)
                    return h1

                def _run(
                    self, x, positive: Optional[int] = None, seed: Optional[int] = None
                ):
                    x2 = x.reshape((-1,))
                    y = np.empty(x2.shape, dtype=np.uint32)
                    for i in range(y.shape[0]):
                        b = x2[i].encode("utf-8")
                        h = MurmurHash3.MurmurHash3_x86_32(b, seed)
                        y[i] = h
                    return (y.reshape(x.shape),)

            ref = ReferenceEvaluator(onx, new_ops=[StringSplit, MurmurHash3])
            got_py = ref.run(None, feeds)
        else:
            got_py = None

        from onnxruntime_extensions import get_library_path

        so = SessionOptions()
        so.register_custom_ops_library(get_library_path())
        sess = InferenceSession(
            onx.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        assert_almost_equal(expected, got[0])

        if ReferenceEvaluator is not None:
            # The pure python implementation does not correctly implement murmurhash3.
            # There are issue with type int.
            assert_almost_equal(expected.shape, got_py[0].shape)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("skl2onnx")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("onnx-extended")
    logger.setLevel(logging.ERROR)

    TestSklearnFeatureHasher().test_feature_hasher_pipeline_list()
    unittest.main(verbosity=2)
