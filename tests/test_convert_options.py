# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from onnxruntime import InferenceSession
from sklearn import __version__ as sklver
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import to_onnx
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from test_utils import TARGET_OPSET


sklver = '.'.join(sklver.split('.')[:2])


class TestConvertOptions(unittest.TestCase):

    @staticmethod
    def get_model_classifiers():
        models = [
            # BaggingClassifier,
            # BernoulliNB,
            # CategoricalNB,
            # CalibratedClassifierCV,
            # ComplementNB,
            DecisionTreeClassifier(max_depth=2),
            # ExtraTreeClassifier,
            # ExtraTreesClassifier,
            # GaussianNB,
            # GaussianProcessClassifier,
            # GradientBoostingClassifier,
            # HistGradientBoostingClassifier,
            # KNeighborsClassifier,
            # LinearDiscriminantAnalysis,
            # LinearSVC,
            LogisticRegression(max_iter=10),
            # LogisticRegressionCV,
            # MLPClassifier,
            # MultinomialNB,
            # NuSVC,
            OneVsRestClassifier(DecisionTreeClassifier(max_depth=2)),
            # PassiveAggressiveClassifier,
            # Perceptron,
            # RandomForestClassifier,
            # SGDClassifier,
            # StackingClassifier,
            # SVC,
            # VotingClassifier,
        ]
        return models

    @staticmethod
    def dict_to_array(proba_as_dict):
        df = DataFrame(proba_as_dict)
        return df.values

    @staticmethod
    def almost_equal(
            expected_label, expected_proba,
            label, probas, zipmap=False, decimal=5):
        if expected_label.tolist() != label.tolist():
            raise AssertionError(
                "Label mismatch %r (expected) != %r." % (
                    expected_label.tolist(),
                    label.tolist()))
        if zipmap:
            raise AssertionError(
                "zipmap should be False, not %r." % zipmap)
        assert_almost_equal(expected_proba, probas, decimal=decimal)

    @staticmethod
    def almost_equal_class_labels(
            expected_label, expected_proba, expected_class_labels,
            label, probas, class_labels,
            zipmap=False, decimal=5):
        if expected_class_labels.tolist() != class_labels.tolist():
            raise AssertionError(
                "Class labels mismatch %r (expected) != %r." % (
                    expected_class_labels.tolist(),
                    class_labels.tolist()))
        if expected_label.tolist() != label.tolist():
            raise AssertionError(
                "Label mismatch %r (expected) != %r." % (
                    expected_label.tolist(),
                    label.tolist()))
        if zipmap:
            raise AssertionError(
                "zipmap should be False, not %r." % zipmap)
        assert_almost_equal(expected_proba, probas, decimal=decimal)

    def classifier_option_output_class_labels(self, use_string):
        data = load_iris()
        X, y = data.data, data.target
        if use_string:
            y = ['cl%d' % _ for _ in y]
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)

        for zipmap, addcl in [(False, True), (False, False)]:
            for cls in TestConvertOptions.get_model_classifiers():
                with self.subTest(cls=cls.__class__.__name__, zipmap=zipmap,
                                  output_class_labels=addcl):
                    cls.fit(X_train, y_train)
                    expected_label = cls.predict(X_test)
                    expected_proba = cls.predict_proba(X_test)

                    onx = to_onnx(
                        cls, X[:1], options={
                            'zipmap': zipmap, 'output_class_labels': addcl},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    if addcl:
                        TestConvertOptions.almost_equal_class_labels(
                            expected_label, expected_proba, cls.classes_,
                            *got, zipmap=zipmap)
                    else:
                        TestConvertOptions.almost_equal(
                            expected_label, expected_proba,
                            *got, zipmap=zipmap)

                    onx = to_onnx(
                        cls, X[:1],
                        options={cls.__class__: {
                            'zipmap': zipmap, 'output_class_labels': addcl}},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    if addcl:
                        TestConvertOptions.almost_equal_class_labels(
                            expected_label, expected_proba, cls.classes_,
                            *got, zipmap=zipmap)
                    else:
                        TestConvertOptions.almost_equal(
                            expected_label, expected_proba,
                            *got, zipmap=zipmap)

                    onx = to_onnx(
                        cls, X[:1],
                        options={id(cls): {
                            'zipmap': zipmap, 'output_class_labels': addcl}},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    if addcl:
                        TestConvertOptions.almost_equal_class_labels(
                            expected_label, expected_proba, cls.classes_,
                            *got, zipmap=zipmap)
                    else:
                        TestConvertOptions.almost_equal(
                            expected_label, expected_proba,
                            *got, zipmap=zipmap)

    @unittest.skipIf(StrictVersion(sklver) < StrictVersion("0.24"),
                     reason="known issue with string")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_classifier_option_output_class_labels_int64(self):
        self.classifier_option_output_class_labels(False)

    @unittest.skipIf(StrictVersion(sklver) < StrictVersion("0.24"),
                     reason="known issue with string")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_classifier_option_output_class_labels_str(self):
        self.classifier_option_output_class_labels(True)

    @staticmethod
    def get_model_multi_label():
        models = [
            MultiOutputClassifier(DecisionTreeClassifier(max_depth=2)),
        ]
        return models

    @staticmethod
    def almost_equal_multi(expected_label, expected_proba, label, *probas,
                           zipmap=False, decimal=5):
        assert_almost_equal(expected_label, label)
        if zipmap == 'columns':
            for row, pr in zip(expected_proba.T, probas):
                assert_almost_equal(
                    row.ravel(), pr.ravel(), decimal=decimal)

        elif zipmap:
            for expected, proba in zip(expected_proba, probas):
                assert_almost_equal(
                    expected_proba,
                    TestConvertOptions.dict_to_array(proba),
                    decimal=decimal)
        else:
            proba = probas[0]
            assert_almost_equal(expected_proba, proba, decimal=decimal)

    @unittest.skipIf(StrictVersion(sklver) < StrictVersion("0.24"),
                     reason="known issue with string")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_multi_label_option_zipmap(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y]).T
        y[0, :] = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for zipmap in [False, True, 'columns']:
            for cls in TestConvertOptions.get_model_multi_label():
                with self.subTest(cls=cls.__class__, zipmap=zipmap):
                    cls.fit(X_train, y_train)
                    expected_label = cls.predict(X_test)
                    expected_proba = cls.predict_proba(X_test)

                    if zipmap == 'columns':
                        # Not implemented.
                        with self.assertRaises(ValueError):
                            to_onnx(cls, X[:1], options={'zipmap': zipmap},
                                    target_opset=TARGET_OPSET)
                        continue

                    onx = to_onnx(cls, X[:1], options={'zipmap': zipmap},
                                  target_opset=TARGET_OPSET)

                    if zipmap:
                        # The converter works but SequenceConstruct
                        # does not support Sequence of Maps.
                        continue

                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    TestConvertOptions.almost_equal_multi(
                        expected_label, expected_proba, *got, zipmap=zipmap)

                    onx = to_onnx(
                        cls, X[:1],
                        options={cls.__class__: {'zipmap': zipmap}},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    assert_almost_equal(expected_label, got[0])

                    onx = to_onnx(
                        cls, X[:1],
                        options={id(cls): {'zipmap': zipmap}},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    assert_almost_equal(expected_label, got[0])

    @staticmethod
    def almost_equal_multi_labels(
            expected_label, expected_proba, expected_class_labels,
            *probas, decimal=5):
        if expected_label.tolist() != probas[0].tolist():
            raise AssertionError(
                "Labels mismatched %r != %r." % (
                    expected_label.tolist(), probas[0].tolist()))
        for pr1, pr2 in zip(expected_proba, probas[1]):
            assert_almost_equal(pr1, pr2, decimal=decimal)
        for la1, la2 in zip(expected_class_labels, probas[2]):
            if la1.tolist() != la2.tolist():
                raise AssertionError(
                    "Class labels mismatched %r != %r." % (
                        la1.tolist(), la2.tolist()))

    @unittest.skipIf(StrictVersion(sklver) < StrictVersion("0.24"),
                     reason="known issue with string")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_multi_label_option_zipmap_class_labels(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y]).T
        y[0, :] = 1
        y[:10, 1] = 3
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for cls in TestConvertOptions.get_model_multi_label():
            with self.subTest(cls=cls.__class__):
                cls.fit(X_train, y_train)
                expected_label = cls.predict(X_test)
                expected_proba = cls.predict_proba(X_test)
                expected_class_labels = [c.classes_ for c in cls.estimators_]
                opts = {'zipmap': False, 'output_class_labels': True}

                onx = to_onnx(cls, X[:1], options=opts,
                              target_opset=TARGET_OPSET)

                sess = InferenceSession(onx.SerializeToString())
                got = sess.run(None, {'X': X_test})
                self.assertEqual(len(got), 3)
                TestConvertOptions.almost_equal_multi_labels(
                    expected_label, expected_proba, expected_class_labels,
                    *got)

                onx = to_onnx(
                    cls, X[:1], options={cls.__class__: opts},
                    target_opset=TARGET_OPSET)
                sess = InferenceSession(onx.SerializeToString())
                got = sess.run(None, {'X': X_test})
                self.assertEqual(len(got), 3)
                TestConvertOptions.almost_equal_multi_labels(
                    expected_label, expected_proba, expected_class_labels,
                    *got)

                onx = to_onnx(
                    cls, X[:1], options={id(cls): opts},
                    target_opset=TARGET_OPSET)
                sess = InferenceSession(onx.SerializeToString())
                got = sess.run(None, {'X': X_test})
                self.assertEqual(len(got), 3)
                TestConvertOptions.almost_equal_multi_labels(
                    expected_label, expected_proba, expected_class_labels,
                    *got)

    @unittest.skipIf(StrictVersion(sklver) < StrictVersion("0.24"),
                     reason="known issue with string")
    @ignore_warnings(category=(FutureWarning, ConvergenceWarning,
                               DeprecationWarning))
    def test_multi_label_option_zipmap_class_labels_string(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        y = numpy.vstack([y, 1 - y]).T
        y[0, :] = 1
        y[:10, 1] = 3
        y = numpy.array(list(map(
            lambda s: "cl%d" % s, y.ravel()))).reshape(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for cls in TestConvertOptions.get_model_multi_label():
            with self.subTest(cls=cls.__class__):
                cls.fit(X_train, y_train)
                expected_label = cls.predict(X_test)
                expected_proba = cls.predict_proba(X_test)
                expected_class_labels = [c.classes_ for c in cls.estimators_]
                opts = {'zipmap': False, 'output_class_labels': True}

                onx = to_onnx(cls, X[:1], options=opts,
                              target_opset=TARGET_OPSET)
                # with open("debugmo2.onnx", "wb") as f:
                #     f.write(onx.SerializeToString())
                sess = InferenceSession(onx.SerializeToString())
                got = sess.run(None, {'X': X_test})
                self.assertEqual(len(got), 3)
                TestConvertOptions.almost_equal_multi_labels(
                    expected_label, expected_proba, expected_class_labels,
                    *got)

                onx = to_onnx(
                    cls, X[:1], options={cls.__class__: opts},
                    target_opset=TARGET_OPSET)
                sess = InferenceSession(onx.SerializeToString())
                got = sess.run(None, {'X': X_test})
                self.assertEqual(len(got), 3)
                TestConvertOptions.almost_equal_multi_labels(
                    expected_label, expected_proba, expected_class_labels,
                    *got)

                onx = to_onnx(
                    cls, X[:1], options={id(cls): opts},
                    target_opset=TARGET_OPSET)
                sess = InferenceSession(onx.SerializeToString())
                got = sess.run(None, {'X': X_test})
                self.assertEqual(len(got), 3)
                TestConvertOptions.almost_equal_multi_labels(
                    expected_label, expected_proba, expected_class_labels,
                    *got)


if __name__ == "__main__":
    # import logging
    # log = logging.getLogger('skl2onnx')
    # log.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestConvertOptions().test_classifier_option_output_class_labels_str()
    unittest.main()
