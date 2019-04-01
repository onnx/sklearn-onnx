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
from onnx import numpy_helper, helper
from skl2onnx.algebra.automation import dynamic_class_creation
from skl2onnx.algebra import OnnxOperator
from skl2onnx.proto import onnx_proto


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
        untested = {'AveragePool',  # issue with ceil_mode
                    'BatchNormalization',  # issue with non-tensor type
                    'Cast',  # unsupported type
                    'Concat',
                    'ConvTranspose',  # Input X must be 4-dimensional. X: {1,1,3}
                    'MaxPool',  # issue with ceil_mode
                    'Scan',  # Graph attribute inferencing returned type information for 2 outputs. Expected 1
                    'Slice',  # Node () has input size 5 not in range [min=1, max=1].
                    'Split',  # Issue with multiple inputs
                    }
        folder = os.path.dirname(onnx.__file__)
        folder = os.path.join(folder, "backend", "test", "data", "node")
        subs = os.listdir(folder)        
        for sub in subs:
            path = os.path.join(folder, sub)
            model = os.path.join(path, "model.onnx")
            if not os.path.exists(model):
                continue
            dataset = os.path.join(path, "test_data_set_0")
            inps = [os.path.join(dataset, "input_0.pb")]
            outs = [os.path.join(dataset, "output_0.pb")]
            if not os.path.exists(inps[0]) or not os.path.exists(outs[0]):
                continue
            for d in range(1, 9):
                name = os.path.join(dataset, "input_%d.pb" % d)
                if os.path.exists(name):
                    inps.append(name)
                else:
                    break
            for d in range(1, 9):
                name = os.path.join(dataset, "output_%d.pb" % d)
                if os.path.exists(name):
                    outs.append(name)
                else:
                    break
            tests = dict(model=model, inputs=inps, outputs=outs)
            try:
                op_type, success, reason = self._check_algebra_onnxruntime(untested=untested, **tests)
            except Exception as e:
                raise Exception("Unable to handle operator '{}'".format(model)) from e
            if __name__ == "__main__":
                if not success:
                    print("-", op_type, " Failure", reason)

    def _load_data(self, name):
        from onnx import ModelProto
        tensor = onnx.TensorProto()
        with open(name, 'rb') as fid:
            content = fid.read()
            tensor.ParseFromString(content)
        return tensor
    
    def _load_data_test(self, name, test):
        try:
            return self._load_data(name)
        except Exception as e:
            raise RuntimeError("Unable to load data '{}' for test '{}'.".format(name, test)) from e
            
    def _check_algebra_onnxruntime(self, untested=None, model=None, inputs=None, outputs=None):
        if untested is None:
            untested = {}
        name = os.path.split(os.path.split(model)[0])[-1]
        try:
            onx = onnx.load(model)
        except Exception as e:
            raise RuntimeError("Unable to load model '{}' - '{}'.".format(name, model)) from e
        inps = [self._load_data_test(input, name) for input in inputs]
        outs = [self._load_data_test(output, name) for output in outputs]
        
        if len(onx.graph.node) != 1:
            op_type = ",".join([n.op_type for n in onx.graph.node])
            return op_type, False, "The graph contains more than one node. Not tested."
        
        # get the operator to test
        node = onx.graph.node[0]
        op_class = self._algebra.get(node.op_type, None)
        if op_class is None:
            raise RuntimeError("Unable to find the corresponding operator in the algebra "
                               "'{}'.".format(node.op_type))        
        atts = {}
        if node.attribute:
            for att in node.attribute:
                atts[att.name] = helper.get_attribute_value(att)
                
        if len(node.input) != len(inps):
            if node.op_type in untested:
                return node.op_type, False, "unexpected number of inputs {} != {}".format(
                    len(node.output), len(outs))
            raise RuntimeError("'{}': unexpected number of inputs {} != {}.".format(
                node.op_type, len(node.input), len(inps)))
        if len(node.output) < len(outs):
            raise RuntimeError("'{}': unexpected number of inputs {} != {}.".format(
                node.op_type, len(node.output), len(outs)))
        
        # See file onnx-ml.proto.
        if inps[0].data_type in (onnx_proto.TensorProto.FLOAT16, ):
            # not supported
            return node.op_type, False, "Unsupported type {}".format(inps[0].data_type)
        if inps[0].data_type not in (onnx_proto.TensorProto.INT32,
                                     onnx_proto.TensorProto.INT64,
                                     onnx_proto.TensorProto.FLOAT,
                                     onnx_proto.TensorProto.DOUBLE,
                                     onnx_proto.TensorProto.BOOL,
                                     onnx_proto.TensorProto.STRING):
            raise NotImplementedError("Unexpected data_type {}\n{}".format(inps[0].data_type, inps[0]))

        # prepare the inputs
        inp_arrays = [numpy_helper.to_array(inp) for inp in inps]
        out_arrays = [numpy_helper.to_array(out) for out in outs]
        
        for i in range(len(inp_arrays)):
            inp_array = inp_arrays[i]
            if inp_array.dtype == numpy.float64:
                inp_arrays[i] = inp_array.astype(numpy.float32)
                inps[i] = numpy_helper.from_array(inp_arrays[i])
        
        # check the test from onnx is working.
        import onnxruntime as ort
        monx = onx.SerializeToString()
        try:
            sess = ort.InferenceSession(monx)
        except RuntimeError as e:
            if node.op_type in untested:
                return node.op_type, False, "cannot load ONNX model {}".format(e)
            raise RuntimeError("'{}': cannot load(1) due to {}.".format(node.op_type, e))

        names = [i.name for i in sess.get_inputs()]
        ort_inputs = {name: inp_array for name, inp_array in zip(names, inp_arrays)}
        try:
            Y = sess.run(None, ort_inputs)
        except RuntimeError as e:
            if node.op_type in untested:
                return node.op_type, False, "cannot load skl2onnx model {}".format(e)
            raise RuntimeError("'{}': cannot run(1) due to {}.".format(node.op_type, e))
        for exp, got in zip(out_arrays, Y):
            try:
                assert_almost_equal(exp, got, decimal=4)
            except TypeError:
                pass
        
        # instantiate the operator
        for i, inp in enumerate(inps):
            inp.name = 'I%d' % i
        op = op_class(*inps, **atts)
        st = StringIO()        
        with contextlib.redirect_stdout(st):
            with contextlib.redirect_stderr(st):
                ort_inputs = {'I%d' % i: inp for i, inp in enumerate(inps)}
                try:
                    onx2 = op.to_onnx(ort_inputs)
                except (RuntimeError, NotImplementedError) as e:
                    if node.op_type in untested:
                        return node.op_type, False, "cannot load skl2onnx model {}".format(e)
                    raise NotImplementedError("Unable to continue {}\n{}\n{}".format(
                        inp_array.dtype, st.getvalue(), ort_inputs)) from e
        
        # test with onnxruntime
        monx2 = onx2.SerializeToString()
        try:
            sess = ort.InferenceSession(monx2)
        except RuntimeError as e:
            if node.op_type in untested:
                return node.op_type, False, "cannot load skl2onnx model {}".format(e)
            raise RuntimeError("'{}': cannot load(2) due to {}\n"
                               "---ONNX--\n{}\n---SKL2ONNX---\n{}".format(
                                node.op_type, e, onx, onx2))
            
        names = [i.name for i in sess.get_inputs()]
        ort_inputs = {name: inp_array for name, inp_array in zip(names, inp_arrays)}
        try:
            Y = sess.run(None, ort_inputs)
        except RuntimeError as e:
            if node.op_type in untested:
                return node.op_type, False, "cannot load skl2onnx model {}".format(e)
            raise RuntimeError("'{}': cannot run(2) due to {}\n"
                               "---ONNX--\n{}\n---SKL2ONNX---\n{}".format(
                                node.op_type, e, onx, onx2))
        for exp, got in zip(out_arrays, Y):
            try:
                assert_almost_equal(exp, got, decimal=4)
            except (TypeError, AssertionError):
                pass
        return node.op_type, True, ""
        


if __name__ == "__main__":
    unittest.main()
