# SPDX-License-Identifier: Apache-2.0

"""
Tests examples from the documentation.
"""
import unittest
from distutils.version import StrictVersion
import os
import sys
import importlib
import subprocess
import onnxruntime


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(
        module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(
                module_name, module_file_path))
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExample(unittest.TestCase):

    def test_documentation_examples(self):

        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, '..', 'examples'))
        found = os.listdir(fold)
        tested = 0
        for name in found:
            if name.startswith("plot_") and name.endswith(".py"):
                if (name == "plot_pipeline_lightgbm.py" and
                        StrictVersion(onnxruntime.__version__) <
                            StrictVersion('1.0.0')):
                    continue
                print("run %r" % name)
                try:
                    mod = import_source(fold, os.path.splitext(name)[0])
                    assert mod is not None
                except FileNotFoundError:
                    # try another way
                    cmds = [sys.executable, "-u",
                            os.path.join(fold, name)]
                    p = subprocess.Popen(
                        cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    res = p.communicate()
                    out, err = res
                    st = err.decode('ascii', errors='ignore')
                    if len(st) > 0 and 'Traceback' in st:
                        if "No such file or directory: 'dot'" in st:
                            # dot not installed, this part
                            # is tested in onnx framework
                            pass
                        elif '"dot" not found in path.' in st:
                            # dot not installed, this part
                            # is tested in onnx framework
                            pass
                        elif ('Please fix either the inputs or '
                                'the model.') in st:
                            # onnxruntime datasets changed in master branch,
                            # still the same in released version on pypi
                            pass
                        elif ('Current official support for domain ai.onnx '
                              'is till opset 12.') in st:
                            # one example is using opset 13 but onnxruntime
                            # only support up to opset 12.
                            pass
                        elif "'str' object has no attribute 'decode'" in st:
                            # unstable bug in scikit-learn<0.24
                            pass
                        else:
                            raise RuntimeError(
                                "Example '{}' (cmd: {} - exec_prefix='{}') "
                                "failed due to\n{}"
                                "".format(name, cmds, sys.exec_prefix, st))
                tested += 1
        if tested == 0:
            raise RuntimeError("No example was tested.")


if __name__ == "__main__":
    unittest.main()
