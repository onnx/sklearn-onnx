import os
import urllib.request
import re
import unittest
import textwrap
from skl2onnx.algebra.automation import dynamic_class_creation
from skl2onnx.algebra import OnnxOperator


class TestMetaOnnx(unittest.TestCase):

    def test_dynamic_class_creation(self):
        res = dynamic_class_creation()
        for cl in res:
            assert hasattr(cl, '__init__')
            assert hasattr(cl, '__doc__')

    def test_mul(self):
        from skl2onnx.algebra.onnx_ops import Mul
        assert Mul.__name__ == 'Mul'
        assert isinstance(Mul('a', 'b'), OnnxOperator)


if __name__ == "__main__":
    unittest.main()
