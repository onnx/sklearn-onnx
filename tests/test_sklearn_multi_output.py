# SPDX-License-Identifier: Apache-2.0

import unittest
from logging import getLogger
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.datasets import load_linnerud, make_multilabel_classification
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


class TestMultiOutputConverter(unittest.TestCase):

    def setUp(self):
        if __name__ == "__main__":
            log = getLogger('skl2onnx')
            log.disabled = True
            # log.setLevel(logging.DEBUG)
            # logging.basicConfig(level=logging.DEBUG)
            pass

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
        X = X.astype(numpy.float32)
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
        onx = to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                      options={id(clf): {'zipmap': False}}, verbose=0)
        self.assertNotIn("ZipMap", str(onx))

        sess = InferenceSession(onx.SerializeToString())
        res = sess.run(None, {'X': X})
        exp_lab = clf.predict(X)
        exp_prb = numpy.transpose(clf.predict_proba(X), (1, 0, 2))
        assert_almost_equal(exp_lab, res[0])
        assert_almost_equal(exp_prb, res[1], decimal=5)


if __name__ == "__main__":
    unittest.main()
