# SPDX-License-Identifier: Apache-2.0

"""
Tests on functions in common.
"""

import unittest
from skl2onnx.common._container import _get_operation_list


class TestAppliedFunctions(unittest.TestCase):
    def test_converters_list(self):
        fcts = _get_operation_list(False)
        assert "Clip" in fcts
        assert isinstance(fcts, dict)
        assert len(fcts) > 15


if __name__ == "__main__":
    unittest.main()
