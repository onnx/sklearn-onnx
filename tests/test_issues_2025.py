# SPDX-License-Identifier: Apache-2.0
import unittest
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class TestInvestigate2025(unittest.TestCase):
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1161_gaussian(self):
        # https://github.com/onnx/sklearn-onnx/issues/1161
        import numpy as np
        import onnx
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
        onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
