# SPDX-License-Identifier: Apache-2.0
import unittest
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class TestInvestigate2025(unittest.TestCase):
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1161_gaussian(self):
        # https://github.com/onnx/sklearn-onnx/issues/1161
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Generate sample data
        X = np.array([[1], [3], [5], [6], [7], [8], [10], [12], [14], [15]])
        y = np.array([3, 2, 7, 8, 7, 6, 9, 11, 10, 12])

        # Define the kernel
        kernel = WhiteKernel()

        # Create and train the Gaussian Process Regressor
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-2
        )
        gpr.fit(X, y)

        # Convert the trained model to ONNX format
        initial_type = [("float_input", FloatTensorType([None, 1]))]
        onnx_model = convert_sklearn(
            gpr,
            initial_types=initial_type,
            options={GaussianProcessRegressor: {"return_std": True}},
        )
        self.assertTrue(onnx_model is not None)


    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1197_elasticnet_dataframe(self):
        # https://github.com/onnx/sklearn-onnx/issues/1197
        # to_onnx fails on ElasticNet when input is a DataFrame with multiple columns
        import numpy as np
        from numpy.testing import assert_allclose
        from pandas import DataFrame
        from sklearn.linear_model import ElasticNet
        from skl2onnx import to_onnx
        from onnxruntime import InferenceSession

        np.random.seed(42)
        n_samples, n_features = 100, 12
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        y = X @ np.random.randn(n_features) + np.random.randn(n_samples)

        col_names = [f"feat_{i}" for i in range(n_features)]
        df = DataFrame(X, columns=col_names)

        lr = ElasticNet(alpha=0.2, l1_ratio=0.2, random_state=42)
        lr.fit(df, y)

        # This should not raise an error
        onx = to_onnx(lr, df)

        # Validate predictions match sklearn
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        inputs = {c: df[[c]].values for c in df.columns}
        ort_pred = sess.run(None, inputs)[0]
        skl_pred = lr.predict(df).reshape(-1, 1)
        assert_allclose(ort_pred, skl_pred, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
