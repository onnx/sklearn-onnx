import unittest
import numpy as np
import onnxruntime as rt

from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


class TestCharVectorizers(unittest.TestCase):
    def _run_vectorizer(self, analyzer):
        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(2, 5),
            min_df=1,
            max_features=1000,
        )
        vec.fit(["купить дрель", "газонокосилка", "пластиковые стяжки"])

        onx = convert_sklearn(
            vec,
            initial_types=[("input", StringTensorType([None, 1]))],
        )

        sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        res = sess.run(
            None,
            {"input": np.array([["купить дрель"]])},
        )
        return res[0]

    def test_char_vectorizer(self):
        out = self._run_vectorizer("char")
        self.assertEqual(len(out.shape), 2)
        self.assertGreater(out.shape[1], 0)

    def test_char_wb_vectorizer(self):
        out = self._run_vectorizer("char_wb")
        self.assertEqual(len(out.shape), 2)
        self.assertGreater(out.shape[1], 0)


if __name__ == "__main__":
    unittest.main()

