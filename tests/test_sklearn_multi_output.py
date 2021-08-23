# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


class TestMultiOutputConverter(unittest.TestCase):
    def test_multi_output_regressor(self):

        X, y = load_linnerud(return_X_y=True)
        clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
        onx = to_onnx(clf, X[:1].astype(numpy.float32),
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32), clf, onx,
            basename="SklearnMultiOutputRegressor")


if __name__ == "__main__":
    unittest.main()
