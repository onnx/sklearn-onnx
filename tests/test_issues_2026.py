# SPDX-License-Identifier: Apache-2.0
import unittest
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class TestInvestigate2026(unittest.TestCase):
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1248_pipeline_subclass(self):
        # https://github.com/onnx/sklearn-onnx/issues/1248
        # Pipeline subclass raises NotImplementedError in skl2onnx >= 1.17.0
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        class MyPipeline(Pipeline):
            pass

        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        pipe = MyPipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        model_onnx = convert_sklearn(
            pipe, initial_types=[("X", FloatTensorType([None, 4]))]
        )
        self.assertIsNotNone(model_onnx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
