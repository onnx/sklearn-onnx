# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.third_party_skl import register_converters
from xgboost import XGBRegressor, XGBClassifier

try:
    from test_utils import dump_single_regression
except ImportError:
    import os
    import sys
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), "..", "tests"))
    from test_utils import dump_single_regression
from test_utils import dump_binary_classification, dump_multiple_classification


class TestXGBoostModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        register_converters()

    def test_xgb_regressor(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBRegressor()
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_single_regression(xgb, suffix="-Dec4")

    def test_xgb_classifier(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0

        xgb = XGBClassifier(n_estimators=3)
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_binary_classification(xgb, label_string=False)

    def test_xgb_classifier_multi(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(
            xgb, allow_failure="StrictVersion(onnx.__version__) "
            "< StrictVersion('1.3.0')")

    def test_xgb_classifier_multi_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier(objective='reg:logistic')
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_multiple_classification(
            xgb, suffix="RegLog",
            allow_failure="StrictVersion(onnx.__version__) < "
            "StrictVersion('1.3.0')")

    def test_xgb_classifier_reglog(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0

        xgb = XGBClassifier(objective='reg:logistic')
        xgb.fit(X, y)
        conv_model = convert_sklearn(
            xgb, initial_types=[
                ('input', FloatTensorType(shape=[None, X.shape[1]]))])
        self.assertTrue(conv_model is not None)
        dump_binary_classification(xgb, suffix="RegLog", label_string=False)


if __name__ == "__main__":
    unittest.main()
