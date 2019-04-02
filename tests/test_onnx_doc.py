import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from skl2onnx.algebra.automation import dynamic_class_creation


class TestOnnxDoc(unittest.TestCase):
    
    def setUp(self):
        self._algebra = dynamic_class_creation()

    def _test_pad(self):
        from skl2onnx.algebra.onnx_ops import Pad        
        
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])        

        pad = Pad('X', output_names=['Y'],
                  mode='constant', value=1.5,
                  pads=[0, 1, 0, 1])

        model_def = pad.to_onnx({'X': X})
        onnx.checker.check_model(model_def)

    def predict_with_onnxruntime(self, model_def, *inputs):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_def.SerializeToString())
        names = [i.name for i in sess.get_inputs()]
        input = {name: input for name, input in zip(names, inputs)}
        res = sess.run(None, input)
        names = [o.name for o in sess.get_outputs()]
        return {name: output for name, output in zip(names, res)}

    def test_transpose2(self):
        from skl2onnx.algebra.onnx_ops import Transpose

        node = Transpose(Transpose('X', perm=[1, 0, 2]),
                         perm=[1, 0, 2], output_names=['Y'])
        X = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

        model_def = node.to_onnx({'X': X})
        onnx.checker.check_model(model_def)
        res = self.predict_with_onnxruntime(model_def, X)
        assert_almost_equal(res['Y'], X)



if __name__ == "__main__":
    unittest.main()
