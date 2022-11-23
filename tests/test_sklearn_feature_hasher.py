# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's feature selection converters
"""
import unittest
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid)
from onnx.checker import check_model
from onnxruntime import InferenceSession
from sklearn.feature_extraction import FeatureHasher
from skl2onnx import to_onnx
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnFeatureHasher(unittest.TestCase):
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
        sess = InferenceSession(onnx_model.SerializeToString())
        feeds = {'X': np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)}
        got = sess.run(None, feeds)
        self.assertEqual(got[0].shape, feeds["X"].shape)
        self.assertEqual(got[0].dtype, feeds["X"].dtype)

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
        sess = InferenceSession(onnx_model.SerializeToString())

        input_strings = ['z0', 'o11', 'd222', 'q4444', 't333', 'c5555']
        as_bytes = [s.encode("utf-8") for s in input_strings]
        feeds = {'X': np.array(as_bytes)}
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

    #def test_generic_univariate_select_int(self):
    #    model = GenericUnivariateSelect()
    #    X = np.array(
    #        [[1, 2, 3, 1], [0, 3, 1, 4], [3, 5, 6, 1], [1, 2, 1, 5]],
    #        dtype=np.int64)
    #    y = np.array([0, 1, 0, 1])
    #    model.fit(X, y)
    #    model_onnx = convert_sklearn(
    #        model, "generic univariate select",
    #        [("input", Int64TensorType([None, X.shape[1]]))],
    #        target_opset=TARGET_OPSET)
    #    self.assertTrue(model_onnx is not None)
    #    dump_data_and_model(
    #        X, model, model_onnx,
    #        basename="SklearnGenericUnivariateSelect")


if __name__ == "__main__":
    unittest.main()
