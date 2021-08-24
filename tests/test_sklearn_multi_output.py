# SPDX-License-Identifier: Apache-2.0

import unittest
from logging import getLogger, DEBUG, basicConfig
import numpy
from sklearn.datasets import load_linnerud, make_multilabel_classification
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


class TestMultiOutputConverter(unittest.TestCase):

    def setUp(self):
        if __name__ == "__main__":
            log = getLogger('skl2onnx')
            log.setLevel(DEBUG)
            basicConfig(level=DEBUG)

    def test_multi_output_regressor(self):

        X, y = load_linnerud(return_X_y=True)
        clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
        onx = to_onnx(clf, X[:1].astype(numpy.float32),
                      target_opset=TARGET_OPSET)
        dump_data_and_model(
            X.astype(numpy.float32), clf, onx,
            basename="SklearnMultiOutputRegressor")

    def test_multi_output_classifier(self):

        X, y = make_multilabel_classification(n_classes=3, random_state=0)
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
        onx = to_onnx(clf, X[:1].astype(numpy.float32),
                      target_opset=TARGET_OPSET,
                      options={id(clf): {'zipmap': False}},
                      verbose=0)
        self.assertNotIn("ZipMap", str(onx))
        dump_data_and_model(
            X.astype(numpy.float32), clf, onx,
            basename="SklearnMultiOutputClassifier")


if __name__ == "__main__":
    cl = TestMultiOutputConverter()
    cl.setUp()
    cl.test_multi_output_classifier()
    unittest.main()
