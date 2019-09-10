import unittest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_multiple_classification,
    dump_multilabel_classification
)


class TestOneVsRestClassifierConverter(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_02(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            first_class=2,
            suffix="F2",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_string(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multiple_classification(
            model,
            verbose=False,
            label_string=True,
            suffix="String",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_multilabel(self):
        model = OneVsRestClassifier(LogisticRegression())
        dump_multilabel_classification(
            model,
            verbose=False,
            suffix="MultiLabel",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.6.0')",
        )
        dump_multilabel_classification(
            model,
            verbose=False,
            label_string=True,
            suffix="MultiLabelString",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.6.0')",
        )
        dump_multilabel_classification(
            model,
            verbose=False,
            first_class=2,
            suffix="MultiLabel",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.6.0')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_ovr_multilabel_reg(self):
        model = OneVsRestClassifier(LinearRegression())
        dump_multilabel_classification(
            model,
            verbose=False,
            suffix="MultiLabelReg-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.6.0')",
        )
        dump_multilabel_classification(
            model,
            verbose=False,
            label_string=False,
            suffix="MultiLabelRegString-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.6.0')",
        )
        dump_multilabel_classification(
            model,
            verbose=False,
            first_class=2,
            suffix="MultiLabelReg-Out0",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.6.0')",
        )


if __name__ == "__main__":
    unittest.main()
