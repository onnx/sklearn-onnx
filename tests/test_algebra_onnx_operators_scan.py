import unittest
import warnings
from distutils.version import StrictVersion
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import pdist, squareform, cdist as scipy_cdist
import onnx
from onnx.onnx_cpp2py_export.checker import ValidationError
from onnxruntime import InferenceSession, __version__ as ort_version
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxAdd, OnnxIdentity, OnnxScan,
    OnnxSub, OnnxReduceSumSquare,
    OnnxSqueeze, OnnxShape
)
try:
    from skl2onnx.algebra.onnx_ops import OnnxConstantOfShape
except ImportError:
    # onnx is too old
    OnnxConstantOfShape = None
from onnx import (
    helper, TensorProto,
    __version__ as onnx__version__
)
from skl2onnx.algebra.complex_functions import (
    onnx_squareform_pdist, onnx_cdist
)


threshold = "0.4.0"


class TestOnnxOperatorsScan(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion(threshold),
                     reason="fails with onnxruntime 0.4.0")
    def test_onnx_example(self):
        sum_in = onnx.helper.make_tensor_value_info(
            'sum_in', onnx.TensorProto.FLOAT, [2])
        next = onnx.helper.make_tensor_value_info(
            'next', onnx.TensorProto.FLOAT, [2])
        sum_out = onnx.helper.make_tensor_value_info(
            'sum_out', onnx.TensorProto.FLOAT, [2])
        scan_out = onnx.helper.make_tensor_value_info(
            'scan_out', onnx.TensorProto.FLOAT, [2])
        add_node = onnx.helper.make_node(
            'Add',
            inputs=['sum_in', 'next'],
            outputs=['sum_out']
        )
        id_node = onnx.helper.make_node(
            'Identity',
            inputs=['sum_out'],
            outputs=['scan_out']
        )
        scan_body = onnx.helper.make_graph(
            [add_node, id_node],
            'scan_body',
            [sum_in, next],
            [sum_out, scan_out]
        )
        node = onnx.helper.make_node(
            'Scan',
            inputs=['initial', 'x'],
            outputs=['y', 'z'],
            num_scan_inputs=1,
            body=scan_body
        )

        initial = helper.make_tensor_value_info(
            'initial', TensorProto.FLOAT, [2, ])
        X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 2])
        Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, ])
        Z = helper.make_tensor_value_info('z', TensorProto.FLOAT, [3, 2])

        graph_def = helper.make_graph(
            [node],
            'test-model',
            [initial, X],
            [Y, Z],
        )

        model_def = helper.make_model(graph_def, producer_name='onnx-example')

        initial = np.array([0, 0]).astype(np.float32).reshape((2,))
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'initial': initial, 'x': x})

        y = np.array([9, 12]).astype(np.float32).reshape((2,))
        z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))
        assert_almost_equal(y, res[0])
        assert_almost_equal(z, res[1])

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion(threshold),
                     reason="fails with onnxruntime 0.4.0")
    def test_onnx_example_algebra(self):
        initial = np.array([0, 0]).astype(np.float32).reshape((2,))
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))

        add_node = OnnxAdd('sum_in', 'next', output_names=['sum_out'])
        id_node = OnnxIdentity(add_node, output_names=['scan_out'])
        scan_body = id_node.to_onnx(
            {'sum_in': initial, 'next': initial},
            outputs=[('sum_out', FloatTensorType()),
                     ('scan_out', FloatTensorType())])

        node = OnnxScan('initial', 'x', output_names=['y', 'z'],
                        num_scan_inputs=1, body=scan_body.graph)
        model_def = node.to_onnx(
            {'initial': initial, 'x': x},
            outputs=[('y', FloatTensorType()),
                     ('z', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'initial': initial, 'x': x})

        y = np.array([9, 12]).astype(np.float32).reshape((2,))
        z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))
        assert_almost_equal(y, res[0])
        assert_almost_equal(z, res[1])

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion(threshold),
                     reason="fails with onnxruntime 0.4.0")
    def test_onnx_example_pdist(self):
        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))

        diff = OnnxSub('next_in', 'next', output_names=['diff'])
        id_next = OnnxIdentity('next_in', output_names=['next_out'])
        norm = OnnxReduceSumSquare(diff, output_names=['norm'], axes=[1])
        flat = OnnxSqueeze(norm, output_names=['scan_out'], axes=[1])
        scan_body = id_next.to_onnx(
            OrderedDict([('next_in', x), ('next', FloatTensorType())]),
            outputs=[('next_out', FloatTensorType([3, 2])),
                     ('scan_out', FloatTensorType([3]))],
            other_outputs=[flat])

        sess = InferenceSession(scan_body.SerializeToString())
        res = sess.run(None, {'next_in': x, 'next': x[:1]})
        assert_almost_equal(x, res[0])
        exp = np.array([0., 18., 20.], dtype=np.float32)
        assert_almost_equal(exp, res[1])

        node = OnnxScan('x', 'x', output_names=['y', 'z'],
                        num_scan_inputs=1, body=scan_body.graph)
        model_def = node.to_onnx({'x': x},
                                 outputs=[('y', FloatTensorType([3, 2])),
                                          ('z', FloatTensorType([3, 3]))])
        try:
            onnx.checker.check_model(model_def)
        except ValidationError as e:
            if StrictVersion(onnx__version__) <= StrictVersion("1.5.0"):
                warnings.warn(e)
            else:
                raise e

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'x': x})

        exp = squareform(pdist(x, metric="sqeuclidean"))
        assert_almost_equal(x, res[0])
        assert_almost_equal(exp, res[1])

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion(threshold),
                     reason="fails with onnxruntime 0.4.0")
    def test_onnx_example_pdist_in(self):
        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_squareform_pdist(cop, dtype=np.float32),
                            output_names=['pdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('pdist', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        assert_almost_equal(exp, res[0])

        x = np.array([1, 2, 4, 5]).astype(np.float32).reshape((2, 2))
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        assert_almost_equal(exp, res[0])

        x = np.array([1, 2, 4, 5, 5, 6]).astype(np.float32).reshape((2, 3))
        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((2, 3))
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = squareform(pdist(x * 2, metric="sqeuclidean"))
        assert_almost_equal(exp, res[0])

    @unittest.skipIf((OnnxConstantOfShape is None or
                      StrictVersion(ort_version) <= StrictVersion(threshold)),
                     reason="fails with onnxruntime 0.4.0")
    def test_onnx_example_constant_of_shape(self):
        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))

        cop2 = OnnxConstantOfShape(OnnxShape('input'), output_names=['mat'])
        model_def = cop2.to_onnx({'input': x},
                                 outputs=[('mat', FloatTensorType())])
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = np.zeros((3, 2), dtype=np.float32)
        assert_almost_equal(exp, res[0])

        tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                               (1,), [-5])
        cop2 = OnnxConstantOfShape(OnnxShape('input'), value=tensor_value,
                                   output_names=['mat'])
        model_def = cop2.to_onnx({'input': x},
                                 outputs=[('mat', FloatTensorType())])
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = np.full((3, 2), -5.)
        assert_almost_equal(exp, res[0])

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion(threshold),
                     reason="fails with onnxruntime 0.4.0")
    def test_onnx_example_cdist_in(self):
        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        x2 = np.array([1.1, 2.1, 4.01, 5.01, 5.001, 4.001, 0, 0]).astype(
            np.float32).reshape((4, 2))
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x2, dtype=np.float32),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = scipy_cdist(x * 2, x2, metric="sqeuclidean")
        assert_almost_equal(exp, res[0], decimal=5)

        x = np.array([[6.1, 2.8, 4.7, 1.2],
                      [5.7, 3.8, 1.7, 0.3],
                      [7.7, 2.6, 6.9, 2.3],
                      [6.0, 2.9, 4.5, 1.5],
                      [6.8, 2.8, 4.8, 1.4],
                      [5.4, 3.4, 1.5, 0.4],
                      [5.6, 2.9, 3.6, 1.3],
                      [6.9, 3.1, 5.1, 2.3]], dtype=np.float32)
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x, dtype=np.float32),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'input': x})
        exp = scipy_cdist(x * 2, x, metric="sqeuclidean")
        assert_almost_equal(exp, res[0], decimal=4)


if __name__ == "__main__":
    unittest.main()
