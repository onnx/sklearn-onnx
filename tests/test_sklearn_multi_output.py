# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
from logging import getLogger
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.datasets import load_linnerud, make_multilabel_classification
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import Ridge, LogisticRegression
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn import __version__ as skl_ver
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET


skl_ver = ".".join(skl_ver.split('.')[:2])


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
    @ignore_warnings(category=(FutureWarning,
                               DeprecationWarning))
    def test_multi_output_classifier(self):
        X, y = make_multilabel_classification(n_classes=3, random_state=0)
        X = X.astype(numpy.float32)
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
        onx = to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                      options={'zipmap': False})
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
                      options={id(clf): {'nocl': True, 'zipmap': False}})
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
                      options={id(clf): {'nocl': False, 'zipmap': False}})
        self.assertNotIn("ZipMap", str(onx))

        sess = InferenceSession(onx.SerializeToString())
        res = sess.run(None, {'X': X})
        exp_lab = clf.predict(X)
        exp_prb = clf.predict_proba(X)
        assert_almost_equal(exp_lab, res[0])
        self.assertEqual(len(exp_prb), len(res[1]))
        for e, g in zip(exp_prb, res[1]):
            assert_almost_equal(e, g, decimal=5)

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available.")
    @unittest.skipIf(StrictVersion(skl_ver) < StrictVersion("0.22"),
                     reason="classes_ attribute is missing")
    @ignore_warnings(category=(FutureWarning,
                               DeprecationWarning))
    def test_multi_output_classifier_exc(self):
        X, y = make_multilabel_classification(n_classes=3, random_state=0)
        X = X.astype(numpy.float32)
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
        clf.classes_ = numpy.array(clf.classes_)
        with self.assertRaises(RuntimeError):
            to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                    options={'zipmap': False, 'output_class_labels': True})

    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="SequenceConstruct not available.")
    @unittest.skipIf(StrictVersion(skl_ver) < StrictVersion("0.22"),
                     reason="classes_ attribute is missing")
    @ignore_warnings(category=(FutureWarning,
                               DeprecationWarning))
    def test_multi_output_classifier_fallback(self):
        X, y = make_multilabel_classification(n_classes=3, random_state=0)
        X = X.astype(numpy.float32)
        clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
        del clf.classes_
        onx = to_onnx(clf, X[:1], target_opset=TARGET_OPSET,
                      options={'zipmap': False, 'output_class_labels': True})
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
