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


class TestOnnxDoc(unittest.TestCase):
    
    def setUp(self):
        self._algebra = dynamic_class_creation()

    def test_pad(self):
        from skl2onnx.algebra.onnx_ops import Pad        
        from onnx import helper
        from onnx import AttributeProto, TensorProto, GraphProto
        
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])        

        pad = Pad('X', output_names=['Y'],
                  mode='constant', value=1.5,
                  pads=[0, 1, 0, 1])

        model_def = pad.to_onnx({'X': X})
        onnx.checker.check_model(model_def)



if __name__ == "__main__":
    unittest.main()
