import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnxruntime import InferenceSession
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxScaler


class TestOnnxOperatorsCascade(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_cascade_add(self):

        def generate_onnx_graph(dim, nbnode, input_name='X1'):
            i1 = input_name
            for i in range(nbnode - 1):
                i2 = (np.ones((1, dim)) * nbnode * 10).astype(np.float32)
                node = OnnxAdd(i1, i2)
                i1 = node
            i2 = (np.ones((1, dim)) * nbnode * 10).astype(np.float32)
            node = OnnxAdd(i1, i2, output_names=['Y'])
            onx = node.to_onnx([(input_name, FloatTensorType((None, dim)))],
                               outputs=[('Y', FloatTensorType())])
            return onx

        exp = [np.array([[11., 11., 11., 11., 11.]]),
               np.array([[42., 42., 42., 42., 42.]]),
               np.array([[93., 93., 93., 93., 93.]]),
               np.array([[100100., 100100., 100100., 100100., 100100.]])]
        for i, nbnode in enumerate((1, 2, 3, 100)):
            onx = generate_onnx_graph(5, nbnode)
            as_string = onx.SerializeToString()
            ort = InferenceSession(as_string)
            X = (np.ones((1, 5)) * nbnode).astype(np.float32)
            res_out = ort.run(None, {'X1': X})
            assert len(res_out) == 1
            res = res_out[0]
            assert_almost_equal(exp[i], res)

        dim = 10
        onx = generate_onnx_graph(dim, 300)
        as_string = onx.SerializeToString()
        ort = InferenceSession(as_string)
        X = (np.ones((1, dim)) * nbnode).astype(np.float32)
        res_out = ort.run(None, {'X1': X})
        assert len(res_out) == 1
        res = res_out[0]
        assert res.shape[1] == dim

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_cascade_scaler(self):

        def generate_onnx_graph(dim, nbnode, input_name='X1'):
            i1 = input_name
            scale = list(np.ones((1, dim)).ravel())
            for i in range(nbnode - 1):
                i2 = list(map(float, np.ones((1, dim)).astype(
                    np.float32).ravel()))
                node = OnnxScaler(i1, offset=i2, scale=scale)
                i1 = node
            i2 = list(map(float, np.ones((1, dim)).astype(np.float32).ravel()))
            node = OnnxScaler(i1, offset=i2, scale=scale, output_names=['Y'])
            onx = node.to_onnx([(input_name, FloatTensorType((None, dim)))],
                               outputs=[('Y', FloatTensorType((None, dim)))])
            return onx

        exp = [np.zeros((1, 5)),
               np.zeros((1, 5)),
               np.zeros((1, 5)),
               np.zeros((1, 5))]
        for i, nbnode in enumerate((1, 2, 3, 100)):
            onx = generate_onnx_graph(5, nbnode)
            as_string = onx.SerializeToString()
            ort = InferenceSession(as_string)
            X = (np.ones((1, 5)) * nbnode).astype(np.float32)
            res_out = ort.run(None, {'X1': X})
            assert len(res_out) == 1
            res = res_out[0]
            assert_almost_equal(exp[i], res)

        dim = 10
        onx = generate_onnx_graph(dim, 300)
        as_string = onx.SerializeToString()
        ort = InferenceSession(as_string)
        X = (np.ones((1, dim)) * nbnode).astype(np.float32)
        res_out = ort.run(None, {'X1': X})
        assert len(res_out) == 1
        res = res_out[0]
        assert res.shape[1] == dim


if __name__ == "__main__":
    unittest.main()
