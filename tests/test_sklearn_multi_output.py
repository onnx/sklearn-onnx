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

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available.")
    def test_multi_output_classifier(self):
        X, y = make_multilabel_classification(n_classes=3, random_state=0)
        X = X.astype(numpy.float32)
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
        with self.assertRaises(NameError):
            to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                    options={id(clf): {'zipmap': False}})
        onx = to_onnx(clf, X[:1], target_opset=TARGET_OPSET)
        self.assertNotIn("ZipMap", str(onx))

        sess = InferenceSession(onx.SerializeToString())
        res = sess.run(None, {'X': X})
        exp_lab = clf.predict(X)
        exp_prb = clf.predict_proba(X)
        assert_almost_equal(exp_lab, res[0])
        self.assertEqual(len(exp_prb), len(res[1]))
        for e, g in zip(exp_prb, res[1]):
            assert_almost_equal(e, g, decimal=5)

        # check option nocl=True
        onx = to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                      options={id(clf): {'nocl': True}})
        self.assertNotIn("ZipMap", str(onx))

        sess = InferenceSession(onx.SerializeToString())
        res = sess.run(None, {'X': X})
        exp_lab = clf.predict(X)
        exp_prb = clf.predict_proba(X)
        assert_almost_equal(exp_lab, res[0])
        self.assertEqual(len(exp_prb), len(res[1]))
        for e, g in zip(exp_prb, res[1]):
            assert_almost_equal(e, g, decimal=5)

        # check option nocl=False
        onx = to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                      options={id(clf): {'nocl': False}})
        self.assertNotIn("ZipMap", str(onx))

        sess = InferenceSession(onx.SerializeToString())
        res = sess.run(None, {'X': X})
        exp_lab = clf.predict(X)
        exp_prb = clf.predict_proba(X)
        assert_almost_equal(exp_lab, res[0])
        self.assertEqual(len(exp_prb), len(res[1]))
        for e, g in zip(exp_prb, res[1]):
            assert_almost_equal(e, g, decimal=5)


if __name__ == "__main__":
    unittest.main()
