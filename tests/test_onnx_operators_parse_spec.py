import os
import urllib.request
import re
import unittest
import textwrap


class TestParseOnnxOperators(unittest.TestCase):

    def test_onnx_operators(self):

        url = 'https://raw.githubusercontent.com/onnx/onnx/master/docs/Operators.md'
        filename = 'Operators.md'
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
            
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        
        reg = "<a href=\\\"#([a-zA-Z0-9]+)\\\">([a-zA-Z0-9]+)</a>"
        reg = re.compile(reg)
        ops = reg.findall(content)
        assert len(ops) > 0

        template = textwrap.dedent("""
            class {0}(OnnxOperator):
                "See `{0} <https://github.com/onnx/onnx/blob/master/docs/Operators.md#{0}>`_."
                pass
        """)
        
        rows = []
        for op in ops:
            rows.append(template.format(op[1]))
        
        with open("onnx_ops.py.new", "w") as f:
            f.write("\n".join(rows))
        
    def test_onnx_operators_ml(self):

        url = 'https://raw.githubusercontent.com/onnx/onnx/master/docs/Operators-ml.md'
        filename = 'Operators-ml.md'
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
            
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        
        reg = "<a href=\\\"#([a-zA-Z0-9.]+)\\\">([a-zA-Z0-9.]+)</a>"
        reg = re.compile(reg)
        ops = reg.findall(content)
        assert len(ops) > 0

        template = textwrap.dedent('''
            class {1}(OnnxOperator):
                """
                Domain is ``ai.onnx.ml``.
                See `{1} <https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#{0}>`_.
                """
                pass
        ''')
        
        rows = []
        for op in ops:
            name = op[1].replace("ai.onnx.ml.", "")
            rows.append(template.format(op[1], name))
        
        with open("onnx_ops_ml.py.new", "w") as f:
            f.write("\n".join(rows))
        


if __name__ == "__main__":
    unittest.main()
