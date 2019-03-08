import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from test_utils import dump_multiple_classification


class TestOneVsRestClassifierConverter(unittest.TestCase):

    def test_ova(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(model,
                                     allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_ova_02(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(model, first_class=2, suffix="F2",
                                     allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_ova_string(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(model, verbose=False, label_string=True, suffix="String",
                                     allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
