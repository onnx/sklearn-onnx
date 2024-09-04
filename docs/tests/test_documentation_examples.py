# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import sys
import importlib
import subprocess
import time
from skl2onnx import __file__ as onnxrt_backend_dev_file

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(onnxrt_backend_dev_file, "..")))


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(unittest.TestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if sys.platform == "win32" else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    if verbose:
                        print(f"failed: {name!r} due to missing dot.")
                    return 0
                raise AssertionError(  # noqa: B904
                    "Example '{}' (cmd: {} - exec_prefix='{}') "
                    "failed due to\n{}"
                    "".format(name, cmds, sys.exec_prefix, st)
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        folds = [
            os.path.normpath(os.path.join(this, "..", "examples")),
            os.path.normpath(os.path.join(this, "..", "tutorial")),
        ]
        for fold in folds:
            found = os.listdir(fold)
            for name in found:
                if not name.endswith(".py") or not name.startswith("plot_"):
                    continue
                reason = None
                if name in {"plot_woe_transformer.py"}:
                    reason = "dot not available"

                if reason:

                    @unittest.skip(reason)
                    def _test_(self, name=name, fold=fold):
                        res = self.run_test(fold, name, verbose=VERBOSE)
                        self.assertTrue(res)

                else:

                    def _test_(self, name=name, fold=fold):
                        res = self.run_test(fold, name, verbose=VERBOSE)
                        self.assertTrue(res)

                short_name = os.path.split(os.path.splitext(name)[0])[-1]
                setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
