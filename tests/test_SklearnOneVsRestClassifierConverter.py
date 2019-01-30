import unittest
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from skl2onnx import convert_sklearn
from test_utils import dump_multiple_classification


class TestOneVsRestClassifierConverter(unittest.TestCase):

    def test_ova(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(model)


if __name__ == "__main__":
    unittest.main()
