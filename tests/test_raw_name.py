import unittest

import numpy
import onnxruntime as rt
from numpy.testing import assert_almost_equal
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from test_utils import TARGET_OPSET


class RawNameTest(unittest.TestCase):

    _raw_names = (
        "float_input",
        "float_input--",
        "float_input(",
        "float_input)",
    )

    @staticmethod
    def _load_data():
        iris = load_iris()
        return iris.data[:, :2], iris.target

    @staticmethod
    def _train_model(X, y):
        return LogisticRegression().fit(X, y)

    @staticmethod
    def _get_initial_types(X, raw_name):
        return [(raw_name, FloatTensorType([None, X.shape[1]]))]

    @staticmethod
    def _predict(clr_onnx, X):
        sess = rt.InferenceSession(clr_onnx.SerializeToString())
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        return sess.run([label_name], {input_name: X.astype(numpy.float32)})[0]

    def test_raw_name(self):
        """
        Assert that input raw names do not break the compilation
        of the graph and that the ONNX model still produces
        correct predictions.
        """
        X, y = self._load_data()
        clr = self._train_model(X, y)
        pred = clr.predict(X)
        for raw_name in self._raw_names:
            with self.subTest(raw_name=raw_name):
                clr_onnx = convert_sklearn(
                    clr,
                    initial_types=self._get_initial_types(X, raw_name),
                    target_opset=TARGET_OPSET)
                pred_onnx = self._predict(clr_onnx, X)
                assert_almost_equal(pred, pred_onnx)


if __name__ == "__main__":
    unittest.main()
