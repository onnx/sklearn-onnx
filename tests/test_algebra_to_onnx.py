import unittest
from distutils.version import StrictVersion
import numpy as np
import onnx
from onnx.defs import onnx_opset_version
from onnxruntime import InferenceSession, __version__ as ort_version
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        InvalidGraph, Fail, InvalidArgument, NotImplemented)
except ImportError:
    InvalidGraph = RuntimeError
    InvalidArgument = RuntimeError
    Fail = RuntimeError
    NotImplemented = RuntimeError
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.linear_model import LinearRegression, LogisticRegression
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxAdd, OnnxLinearRegressor, OnnxIdentity)
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.proto import get_latest_tested_opset_version
from test_utils import TARGET_OPSET


ort_version = ort_version.split('+')[0]


class TestOnnxOperatorsToOnnx(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_ml(self):
        def generate_onnx_graph(opv):
            node = OnnxAdd(('X1', FloatTensorType()),
                           np.array([0.1], dtype=np.float32),
                           op_version=opv)
            out = OnnxLinearRegressor(
                node, coefficients=[0.3, 0.3, 0.4, 0.5, 0.6],
                intercepts=[-50.], op_version=1)
            last = OnnxIdentity(out, output_names=['Y'], op_version=opv)
            onx = last.to_onnx([('X1', FloatTensorType((None, 5)))],
                               outputs=[('Y', FloatTensorType())],
                               target_opset=opv)
            return onx, (node, out, last)

        for opv in [{'': 10}] + list(range(9, TARGET_OPSET + 1)):
            with self.subTest(opv=opv):
                if isinstance(opv, dict):
                    if opv[''] > get_latest_tested_opset_version():
                        continue
                elif (opv is not None and
                        opv > get_latest_tested_opset_version()):
                    continue
                for i, nbnode in enumerate((1, 2, 3, 100)):
                    onx, nodes = generate_onnx_graph(opv=opv)
                    if opv == {'': 10}:
                        for im in onx.opset_import:
                            if im.version > 10:
                                raise AssertionError(
                                    "Wrong final opset\nopv={}\n{}".format(
                                        opv, onx))
                    else:
                        for im in onx.opset_import:
                            if im.version > opv:
                                raise AssertionError(
                                    "Wrong final opset\nopv={}\n{}".format(
                                        opv, onx))
                    as_string = onx.SerializeToString()
                    try:
                        ort = InferenceSession(as_string)
                    except (InvalidGraph, InvalidArgument) as e:
                        if (isinstance(opv, dict) and
                                opv[''] >= onnx_opset_version()):
                            continue
                        if (isinstance(opv, int) and
                                opv >= onnx_opset_version()):
                            continue
                        raise AssertionError(
                            "Unable to load opv={}\n---\n{}\n---".format(
                                opv, onx)) from e
                    X = (np.ones((1, 5)) * nbnode).astype(np.float32)
                    res_out = ort.run(None, {'X1': X})
                    assert len(res_out) == 1
                    res = res_out[0]
                    self.assertEqual(res.shape, (1, 1))
                    inputs = None
                    expected = [[('Ad_C0', FloatTensorType(shape=[]))],
                                [('Li_Y0', FloatTensorType(shape=[]))],
                                [('Y', FloatTensorType(shape=[]))]]
                    for i, node in enumerate(nodes):
                        shape = node.get_output_type_inference(inputs)
                        self.assertEqual(len(shape), 1)
                        if isinstance(shape[0], tuple):
                            self.assertEqual(str(expected[i]), str(shape))
                        else:
                            self.assertEqual(
                                str(expected[i]),
                                str([(shape[0].onnx_name, shape[0].type)]))
                        inputs = shape

    def common_test_sub_graph(self, first_input, model, options=None,
                              cls_type=FloatTensorType, start=9):
        def generate_onnx_graph(opv):
            dtype = np.float32 if cls_type == FloatTensorType else np.float64
            node = OnnxAdd(first_input, np.array([0.1], dtype=dtype),
                           op_version=opv)
            lr = model()
            lr.fit(np.ones([10, 5]), np.arange(0, 10) % 3)
            out = OnnxSubEstimator(lr, node, op_version=1, options=options)
            if model == LogisticRegression:
                last = OnnxIdentity(out[1], output_names=['Y'], op_version=opv)
            else:
                last = OnnxIdentity(out, output_names=['Y'], op_version=opv)
            onx = last.to_onnx([('X1', cls_type((None, 5)))],
                               outputs=[('Y', cls_type())],
                               target_opset=opv)
            return onx

        dtype = np.float32 if cls_type == FloatTensorType else np.float64

        opsets = list(range(start, TARGET_OPSET + 1))
        for opv in [{'': TARGET_OPSET}] + opsets:
            with self.subTest(opv=opv):
                if isinstance(opv, dict):
                    if opv[''] > get_latest_tested_opset_version():
                        continue
                elif (opv is not None and
                        opv > get_latest_tested_opset_version()):
                    continue
                for i, nbnode in enumerate((1, 2, 3, 100)):
                    onx = generate_onnx_graph(opv=opv)
                    if opv == {'': TARGET_OPSET}:
                        for im in onx.opset_import:
                            if im.version > TARGET_OPSET:
                                raise AssertionError(
                                    "Wrong final opset\nopv={}\n{}".format(
                                        opv, onx))
                    else:
                        for im in onx.opset_import:
                            if im.version > opv:
                                raise AssertionError(
                                    "Wrong final opset\nopv={}\n{}".format(
                                        opv, onx))
                    self.assertNotIn('zipmap', str(onx).lower())
                    as_string = onx.SerializeToString()
                    try:
                        ort = InferenceSession(as_string)
                    except (InvalidGraph, InvalidArgument, Fail,
                            NotImplemented) as e:
                        if (isinstance(opv, dict) and
                                opv[''] >= onnx_opset_version()):
                            continue
                        if (isinstance(opv, int) and
                                opv >= onnx_opset_version()):
                            continue
                        raise AssertionError(
                            "Unable to load opv={}\n---\n{}\n---".format(
                                opv, onx)) from e
                    X = (np.ones((1, 5)) * nbnode).astype(dtype)
                    res_out = ort.run(None, {'X1': X})
                    assert len(res_out) == 1
                    res = res_out[0]
                    if model == LogisticRegression:
                        self.assertEqual(res.shape, (1, 3))
                    else:
                        self.assertEqual(res.shape, (1, 1))

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_tuple(self):
        self.common_test_sub_graph(
            ('X1', FloatTensorType()), LinearRegression)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.4.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_tuple_double(self):
        self.common_test_sub_graph(
            ('X1', DoubleTensorType()), LinearRegression,
            cls_type=DoubleTensorType)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_str(self):
        self.common_test_sub_graph('X1', LinearRegression)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.4.0"),
        reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_str_double(self):
        self.common_test_sub_graph('X1', LinearRegression,
                                   cls_type=DoubleTensorType)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_tuple_cls(self):
        self.common_test_sub_graph(
            ('X1', FloatTensorType()), LogisticRegression,
            {'zipmap': False})

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.4.0"),
        reason="not available")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.10.0"),
        reason="ArgMax not available for double")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_tuple_cls_double(self):
        self.common_test_sub_graph(
            ('X1', DoubleTensorType()), LogisticRegression,
            options={'zipmap': False}, cls_type=DoubleTensorType,
            start=13)

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_str_cls(self):
        self.common_test_sub_graph('X1', LogisticRegression,
                                   {'zipmap': False})

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.4.0"),
        reason="not available")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.10.0"),
        reason="ArgMax not available for double")
    @ignore_warnings(category=DeprecationWarning)
    def test_sub_graph_str_cls_double(self):
        self.common_test_sub_graph(
            'X1', LogisticRegression, options={'zipmap': False},
            cls_type=DoubleTensorType, start=13)


if __name__ == "__main__":
    unittest.main()
