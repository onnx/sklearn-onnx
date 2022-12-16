# SPDX-License-Identifier: Apache-2.0
# coding: utf-8
"""
Tests scikit-learn's feature selection converters
"""
import unittest
import packaging.version as pv
import numpy as np
from pandas import DataFrame
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node,
    make_graph, make_tensor_value_info, make_opsetid)
from onnx.checker import check_model
from onnxruntime import __version__ as ort_version
from onnxruntime import InferenceSession
from sklearn.feature_extraction import FeatureHasher
from skl2onnx import to_onnx
from skl2onnx.common.data_types import (
    StringTensorType, Int64TensorType, FloatTensorType,
    DoubleTensorType)
from test_utils import TARGET_OPSET


class TestSklearnFeatureHasher(unittest.TestCase):

    @unittest.skipIf(pv.Version(ort_version) < pv.Version("1.12.0"),
                     reason="no murmurhash3 in ort")
    def test_ort_murmurhash3_int(self):
        X = make_tensor_value_info('X', TensorProto.UINT32, [None])
        Y = make_tensor_value_info('Y', TensorProto.UINT32, [None])
        node = make_node('MurmurHash3', ['X'], ['Y'], domain="com.microsoft",
                         positive=1, seed=0)
        graph = make_graph([node], 'hash', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[
            make_opsetid('', TARGET_OPSET),
            make_opsetid('com.microsoft', 1)])
        check_model(onnx_model)
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"])
        feeds = {'X': np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)}
        got = sess.run(None, feeds)
        self.assertEqual(got[0].shape, feeds["X"].shape)
        self.assertEqual(got[0].dtype, feeds["X"].dtype)

    @unittest.skipIf(pv.Version(ort_version) < pv.Version("1.12.0"),
                     reason="no murmurhash3 in ort")
    def test_ort_murmurhash3_string(self):
        X = make_tensor_value_info('X', TensorProto.STRING, [None])
        Y = make_tensor_value_info('Y', TensorProto.INT32, [None])
        node = make_node('MurmurHash3', ['X'], ['Y'], domain="com.microsoft",
                         positive=0, seed=0)
        graph = make_graph([node], 'hash', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[
            make_opsetid('', TARGET_OPSET),
            make_opsetid('com.microsoft', 1)])
        check_model(onnx_model)
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"])

        input_strings = ['z0', 'o11', 'd222', 'q4444', 't333', 'c5555']
        feeds = {'X': np.array(input_strings)}
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

        skl = FeatureHasher(n_features, input_type='string', dtype=np.uint32)
        expected = skl.transform(feeds["X"].reshape((-1, 1)))
        dense = expected.todense()
        for i, (a, b) in enumerate(zip(dense.tolist(), mat.tolist())):
            if a != b:
                raise AssertionError(f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher(self):
        n_features = 5
        input_strings = ['z0', 'o11', 'd222', 'q4444', 't333', 'c5555']
        data = np.array(input_strings).reshape((-1, 1))
        for alternate_sign, dtype in [(True, np.float32),
                                      (True, np.float64),
                                      (True, np.int64),
                                      (False, np.float32)]:
            if dtype == np.float32:
                final_type = FloatTensorType
            elif dtype == np.float64:
                final_type = DoubleTensorType
            elif dtype in (np.int32, np.uint32, np.int64):
                final_type = Int64TensorType
            else:
                final_type = None
            with self.subTest(alternate_sign=alternate_sign, dtype=dtype):
                model = FeatureHasher(n_features=n_features,
                                      alternate_sign=alternate_sign,
                                      dtype=dtype,
                                      input_type='string')
                model.fit(data)
                expected = model.transform(data).todense()

                model_onnx = to_onnx(
                    model, initial_types=[("X", StringTensorType([None, 1]))],
                    target_opset=TARGET_OPSET,
                    final_types=[('Y', final_type([None, 1]))])
                sess = InferenceSession(
                    onnx_model.SerializeToString(),
                    providers=["CPUExecutionProvider"])
                got = sess.run(None, {'X': data})
                self.assertEqual(expected.shape, got[0].shape)
                self.assertEqual(expected.dtype, got[0].dtype)
                for i, (a, b) in enumerate(zip(expected.tolist(),
                                               got[0].tolist())):
                    if a != b:
                        raise AssertionError(
                            f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_two_columns(self):
        n_features = 5
        input_strings = ['z0', 'o11', 'd222', 'q4444', 't333', 'c5555']
        data = np.array(input_strings).reshape((-1, 2))

        model = FeatureHasher(n_features=n_features,
                              alternate_sign=True,
                              dtype=np.float32,
                              input_type='string')
        model.fit(data)
        expected = model.transform(data).todense()

        model_onnx = to_onnx(
            model, initial_types=[
                ("X", StringTensorType([None, data.shape[1]]))],
            target_opset=TARGET_OPSET,
            final_types=[('Y', FloatTensorType([None, n_features]))])
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"])
        got = sess.run(None, {'X': data})
        self.assertEqual(expected.shape, got[0].shape)
        self.assertEqual(expected.dtype, got[0].dtype)
        for i, (a, b) in enumerate(zip(expected.tolist(),
                                       got[0].tolist())):
            if a != b:
                raise AssertionError(
                    f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_dataframe(self):
        n_features = 5
        input_strings = ['z0', 'o11', 'd222', 'q4444', 't333', 'c5555']
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

        model = FeatureHasher(n_features=n_features,
                              alternate_sign=True,
                              dtype=np.float32,
                              input_type='string')
        model.fit(data)
        expected = model.transform(data).todense()
        print(expected)

        model_onnx = to_onnx(
            model, initial_types=[
                ("X", StringTensorType([None, data.shape[0]]))],
            target_opset=TARGET_OPSET,
            final_types=[('Y', FloatTensorType([None, n_features]))])
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"])
        got = sess.run(None, {'X': data_nx})
        self.assertEqual(expected.shape, got[0].shape)
        self.assertEqual(expected.dtype, got[0].dtype)
        for i, (a, b) in enumerate(zip(expected.tolist(),
                                       got[0].tolist())):
            if a != b:
                raise AssertionError(
                    f"Discrepancies at line {i}: {a} != {b}")

    def test_feature_hasher_two_columns_unicode(self):
        n_features = 5
        input_strings = ['z0', 'o11', 'd222', '고리', 'é', 'ô']
        data = np.array(input_strings).reshape((-1, 2))

        model = FeatureHasher(n_features=n_features,
                              alternate_sign=True,
                              dtype=np.float32,
                              input_type='string')
        model.fit(data)
        expected = model.transform(data).todense()

        model_onnx = to_onnx(
            model, initial_types=[
                ("X", StringTensorType([None, data.shape[1]]))],
            target_opset=TARGET_OPSET,
            final_types=[('Y', FloatTensorType([None, n_features]))])
        sess = InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"])
        got = sess.run(None, {'X': data})
        self.assertEqual(expected.shape, got[0].shape)
        self.assertEqual(expected.dtype, got[0].dtype)
        for i, (a, b) in enumerate(zip(expected.tolist(),
                                       got[0].tolist())):
            if a != b:
                raise AssertionError(
                    f"Discrepancies at line {i}: {a} != {b}")


if __name__ == "__main__":
    unittest.main()
