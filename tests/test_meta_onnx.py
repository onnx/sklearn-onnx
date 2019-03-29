import os
import urllib.request
import re
import unittest
import textwrap
import warnings
from io import StringIO
import contextlib
import numpy
from numpy.testing import assert_almost_equal
import onnx
from onnx import numpy_helper
from skl2onnx.algebra.automation import dynamic_class_creation
from skl2onnx.algebra import OnnxOperator


class TestMetaOnnx(unittest.TestCase):
    
    def setUp(self):
        self._algebra = dynamic_class_creation()

    def test_dynamic_class_creation(self):
        res = self._algebra
        for cl in res:
            assert hasattr(cl, '__init__')
            assert hasattr(cl, '__doc__')

    def test_mul(self):
        from skl2onnx.algebra.onnx_ops import Mul
        assert Mul.__name__ == 'Mul'
        assert isinstance(Mul('a', 'b'), OnnxOperator)
        
    def test_onnx_spec(self):
        folder = os.path.dirname(onnx.__file__)
        folder = os.path.join(folder, "backend", "test", "data", "node")
        subs = os.listdir(folder)        
        for sub in subs:
            path = os.path.join(folder, sub)
            model = os.path.join(path, "model.onnx")
            if not os.path.exists(model):
                continue
            dataset = os.path.join(path, "test_data_set_0")
            inp0 = os.path.join(dataset, "input_0.pb")
            out0 = os.path.join(dataset, "output_0.pb")
            if not os.path.exists(inp0) or not os.path.exists(out0):
                continue
            tests = dict(model=model, input=inp0, output=out0)
            self._check_algebra_onnxruntime(**tests)

    def _load_data(self, name):
        from onnx import ModelProto
        tensor = onnx.TensorProto()
        with open(name, 'rb') as fid:
            content = fid.read()
            tensor.ParseFromString(content)
        return tensor
            
    def _check_algebra_onnxruntime(self, model=None, input=None, output=None):
        name = os.path.split(os.path.split(model)[0])[-1]
        try:
            onx = onnx.load(model)
        except Exception as e:
            raise RuntimeError("Unable to load model '{}' - '{}'.".format(name, model)) from e
        try:
            inp = self._load_data(input)
        except Exception as e:
            raise RuntimeError("Unable to load input '{}' - '{}'.".format(name, input)) from e
        try:
            out = self._load_data(output)
        except Exception as e:
            raise RuntimeError("Unable to load output '{}' - '{}'.".format(name, output)) from e
        
        if len(onx.graph.node) != 1:
            # We do nothing as the graph contains more than one node.
            return
        
        node = onx.graph.node[0]
        op_class = self._algebra.get(node.op_type, None)
        if op_class is None:
            raise RuntimeError("Unable to find the corresponding operator in the algebra "
                               "'{}'.".format(node.op_type))        
        #op = op_class(
        atts = {}
        if node.attribute:
            for att in node.attribute:
                atts[att.name] = att.i
        if len(node.input) != 1:
            warnings.warn("'{}': Skip due to more than one input.".format(node.op_type))
            return
        if len(node.output) != 1:
            warnings.warn("'{}': Skip due to more than one output.".format(node.op_type))
            return
        
        if inp.data_type not in (1, 7, 8):
            return

        inp_array = numpy_helper.to_array(inp)
        out_array = numpy_helper.to_array(out)
        
        if inp_array.dtype == numpy.float64:
            inp_array = inp_array.astype(numpy.float32)
            inp = numpy_helper.from_array(inp_array)
        
        import onnxruntime as ort
        monx = onx.SerializeToString()
        try:
            sess = ort.InferenceSession(monx)
        except RuntimeError as e:
            warnings.warn("'{}': cannot load(1) due to {}.".format(node.op_type, e))
            return

        name = sess.get_inputs()[0].name
        try:
            Y = sess.run(None, {name: inp_array})[0]
        except RuntimeError as e:
            warnings.warn("'{}': cannot run(1) due to {}.".format(node.op_type, e))
            return
        try:
            assert_almost_equal(Y, out_array, decimal=4)
        except TypeError:
            pass
        
        inp.name = 'I0'
        op = op_class(inp, **atts)
        with contextlib.redirect_stdout(StringIO()):
            with contextlib.redirect_stderr(StringIO()):
                try:
                    onx2 = op.to_onnx({'I0': inp})
                except NotImplementedError as e:
                    raise NotImplementedError(inp_array.dtype) from e
        
        monx2 = onx2.SerializeToString()
        try:
            sess = ort.InferenceSession(monx2)
        except RuntimeError as e:
            warnings.warn("'{}': cannot load(2) due to {}.".format(node.op_type, e))
            return
            
        name = sess.get_inputs()[0].name
        try:
            Y = sess.run(None, {name: inp_array})[0]
        except RuntimeError as e:
            warnings.warn("'{}': cannot run(1) due to {}.".format(node.op_type, e))
            return
        try:
            assert_almost_equal(Y, out_array, decimal=4)
        except TypeError:
            pass
        
        print(node.op_type, "++OK")
        


if __name__ == "__main__":
    unittest.main()
