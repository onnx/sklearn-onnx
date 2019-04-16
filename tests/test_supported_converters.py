"""
Tests scikit-learn's binarizer converter.
"""

import unittest
from skl2onnx import supported_converters


class TestSupportedConverters(unittest.TestCase):
    def test_converters_list(self):
        names = supported_converters(False)
        assert "SklearnBernoulliNB" in names
        assert len(names) > 35

    def test_sklearn_converters(self):
        names = supported_converters(True)
        assert "BernoulliNB" in names
        assert len(names) > 35


if __name__ == "__main__":
    unittest.main()
