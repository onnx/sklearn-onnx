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
    def almost_equal_class_labels(
            expected_label, expected_proba, expected_class_labels,
            label, probas, class_labels,
            zipmap=False, decimal=5):
        assert_almost_equal(expected_class_labels, class_labels)
        assert_almost_equal(expected_label, label)
        if zipmap:
            raise AssertionError(
                "zipmap should be False, not %r." % zipmap)
        proba = probas[0]
        assert_almost_equal(expected_proba, proba, decimal=decimal)

    def classifier_option_output_class_labels(self, use_string):
        data = load_iris()
        X, y = data.data, data.target
        if use_string:
            y = ['cl%d' % _ for _ in y]
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

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
                    TestConvertOptions.almost_equal_class_labels(
                        expected_label, expected_proba, cls.classes_,
                        *got, zipmap=zipmap)

                    onx = to_onnx(
                        cls, X[:1],
                        options={cls.__class__: {
                            'zipmap': zipmap, 'output_class_labels': addcl}},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    TestConvertOptions.almost_equal_class_labels(
                        expected_label, expected_proba, cls.classes_,
                        *got, zipmap=zipmap)

                    onx = to_onnx(
                        cls, X[:1],
                        options={id(cls): {
                            'zipmap': zipmap, 'output_class_labels': addcl}},
                        target_opset=TARGET_OPSET)
                    sess = InferenceSession(onx.SerializeToString())
                    got = sess.run(None, {'X': X_test})
                    TestConvertOptions.almost_equal_class_labels(
                        expected_label, expected_proba, cls.classes_,
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


if __name__ == "__main__":
    # import logging
    # log = logging.getLogger('skl2onnx')
    # log.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestConvertOptions().test_multi_label_option_zipmap()
    unittest.main()
