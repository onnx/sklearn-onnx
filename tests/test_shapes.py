# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import numpy
import onnx
import onnxruntime as rt
from onnxruntime import __version__ as ort_version
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import TARGET_OPSET


ort_version = ort_version.split('+')[0]


class TestShapes(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.6.0"),
                     reason="not available")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.0.0"),
                     reason="not available")
    def test_onnxruntime_shapes_reg(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr = RandomForestRegressor(max_depth=1)
        clr.fit(X_train, y_train)
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onx = convert_sklearn(clr, initial_types=initial_type,
                              target_opset=TARGET_OPSET)
        sess = rt.InferenceSession(onx.SerializeToString())
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})
        shape1 = sess.get_inputs()[0].shape
        shape2 = sess.get_outputs()[0].shape
        ishape = onnx.shape_inference.infer_shapes(onx)
        dims = ishape.graph.output[0].type.tensor_type.shape.dim
        oshape = [d.dim_value for d in dims]
        assert shape1 == [None, 4]
        assert shape2 == [None, 1]
        assert oshape == [0, 1]
        assert pred_onx[0].shape[1] == shape2[1]

    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion("1.6.0"),
                     reason="not available")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("1.0.0"),
                     reason="not available")
    def test_onnxruntime_shapes_clr(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr = RandomForestClassifier(max_depth=1)
        clr.fit(X_train, y_train)
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onx = convert_sklearn(clr, initial_types=initial_type,
                              options={id(clr): {'zipmap': False}},
                              target_opset=TARGET_OPSET)
        sess = rt.InferenceSession(onx.SerializeToString())
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})
        shape1 = sess.get_inputs()[0].shape
        shape2 = sess.get_outputs()[0].shape
        assert shape1 == [None, 4]
        assert shape2 in ([None, 1], [1], [None])
        if len(pred_onx[0].shape) > 1:
            assert pred_onx[0].shape[1] == shape2[1]

        try:
            ishape = onnx.shape_inference.infer_shapes(onx)
        except RuntimeError:
            # Shape inference does not work?
            ishape = None
        if ishape is None:
            oshape = None
        else:
            dims = ishape.graph.output[0].type.tensor_type.shape.dim
            oshape = [d.dim_value for d in dims]
            self.assertIn(oshape, (None, [0]))
            dims = ishape.graph.output[1].type.tensor_type.shape.dim
            oshape = [d.dim_value for d in dims]
            self.assertIn(oshape, (None, [0, 3]))


if __name__ == "__main__":
    unittest.main()
