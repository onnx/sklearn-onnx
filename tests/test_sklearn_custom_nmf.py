# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.decomposition import NMF
import onnx
from skl2onnx.common.data_types import FloatTensorType, onnx_built_with_ml
from skl2onnx.algebra.onnx_ops import (
    OnnxArrayFeatureExtractor, OnnxMul, OnnxReduceSum)
from onnxruntime import InferenceSession
from test_utils import TARGET_OPSET


class TestSklearnCustomNMF(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.4.1"),
                     reason="not available")
    def test_custom_nmf(self):

        mat = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        mat[:mat.shape[1], :] += np.identity(mat.shape[1])

        mod = NMF(n_components=2, max_iter=2)
        W = mod.fit_transform(mat)
        H = mod.components_

        def predict(W, H, row_index, col_index):
            return np.dot(W[row_index, :], H[:, col_index])

        pred = mod.inverse_transform(W)

        exp = []
        got = []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                exp.append((i, j, pred[i, j]))
                got.append((i, j, predict(W, H, i, j)))

        max_diff = max(abs(e[2] - o[2]) for e, o in zip(exp, got))
        assert max_diff <= 1e-5

        def nmf_to_onnx(W, H):
            """
            The function converts a NMF described by matrices
            *W*, *H* (*WH* approximate training data *M*).
            into a function which takes two indices *(i, j)*
            and returns the predictions for it. It assumes
            these indices applies on the training data.
            """
            col = OnnxArrayFeatureExtractor(H, 'col')
            row = OnnxArrayFeatureExtractor(W.T, 'row')
            dot = OnnxMul(col, row, op_version=TARGET_OPSET)
            res = OnnxReduceSum(dot, output_names="rec",
                                op_version=TARGET_OPSET)
            indices_type = np.array([0], dtype=np.int64)
            onx = res.to_onnx(inputs={'col': indices_type,
                                      'row': indices_type},
                              outputs=[('rec', FloatTensorType((None, 1)))])
            return onx

        model_onnx = nmf_to_onnx(W.astype(np.float32),
                                 H.astype(np.float32))
        sess = InferenceSession(model_onnx.SerializeToString())

        def predict_onnx(sess, row_indices, col_indices):
            res = sess.run(None,
                           {'col': col_indices,
                            'row': row_indices})
            return res

        onnx_preds = []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                row_indices = np.array([i], dtype=np.int64)
                col_indices = np.array([j], dtype=np.int64)
                pred = predict_onnx(sess, row_indices, col_indices)[0]
                onnx_preds.append((i, j, pred[0, 0]))

        max_diff = max(abs(e[2] - o[2]) for e, o in zip(exp, onnx_preds))
        assert max_diff <= 1e-5


if __name__ == "__main__":
    unittest.main()
