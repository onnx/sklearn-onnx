"""Tests GLMRegressor converter."""

import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import BaggingClassifier
# Requires PR #488.
# from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
try:
    from sklearn.ensemble import VotingClassifier
except ImportError:
    VotingClassifier = None
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
try:
    from sklearn.naive_bayes import ComplementNB
except ImportError:
    ComplementNB = None
try:
    from sklearn.ensemble import StackingClassifier
except ImportError:
    StackingClassifier = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_classification_model)  # , TARGET_OPSET)

TARGET_OPSET = 12  # change when PR 551
warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


class TestSklearnDoubleTensorTypeClassifier(unittest.TestCase):

    def _common_classifier(
            self, model_cls_set, name_root=None, debug=False,
            raw_scores=True, pos_features=False, is_int=False):
        for model_cls in model_cls_set:
            if name_root is None:
                name = model_cls.__name__
            else:
                name = name_root
            for n_cl in [2, 3]:
                model, X = fit_classification_model(
                    model_cls(), n_cl, n_features=4,
                    pos_features=pos_features, is_int=is_int)
                pmethod = ('decision_function_binary' if n_cl == 2 else
                           'decision_function')
                bs = [True, False] if raw_scores else [False]
                for b in bs:
                    for z in [False]:
                        # zipmap does not allow tensor(double) as inputs
                        with self.subTest(n_classes=n_cl, raw_scores=b,
                                          model=name):
                            if raw_scores:
                                options = {"raw_scores": b,
                                           "zipmap": z}
                            else:
                                options = {"zipmap": z}
                            model_onnx = convert_sklearn(
                                model, "model",
                                [("input", DoubleTensorType(
                                    [None, X.shape[1]]))],
                                target_opset=TARGET_OPSET,
                                options={id(model): options})
                            if debug:
                                print(model_onnx)
                            self.assertIn("elem_type: 11", str(model_onnx))
                            methods = None if not b else ['predict', pmethod]
                            if not b and n_cl == 2:
                                # onnxruntime does not support sigmoid for
                                # DoubleTensorType
                                continue
                            dump_data_and_model(
                                X.astype(np.float64), model, model_onnx,
                                methods=methods,
                                basename="Sklearn{}Double2RAW{}"
                                         "ZIP{}CL{}".format(
                                            name,
                                            1 if b else 0,
                                            1 if z else 0, n_cl))

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.5.0"),
        reason="ArgMax is missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_model_logistic_64(self):
        self._common_classifier([LogisticRegression])

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.5.0"),
        reason="ArgMax is missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_modelsgd_64(self):
        self._common_classifier([SGDClassifier])
        self._common_classifier([lambda: SGDClassifier(loss='hinge')],
                                "SGDClassifierHinge")
        self._common_classifier([lambda: SGDClassifier(loss='perceptron')],
                                "SGDClassifierPerceptron")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Reciprocal are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_modelsgdlog_64(self):
        self._common_classifier([lambda: SGDClassifier(loss='log')],
                                "SGDClassifierLog")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Relu are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_mlpclassifier_relu_64(self):
        self._common_classifier(
            [lambda: MLPClassifier(activation='Relu')],
            "MLPClassifierRelu", raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Tanh are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_mlpclassifier_tanh_64(self):
        self._common_classifier(
            [lambda: MLPClassifier(activation='tanh',
                                   hidden_layer_sizes=(2,))],
            "MLPClassifierTanh", raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Sigmoid are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_mlpclassifier_logistic_64(self):
        self._common_classifier(
            [lambda: MLPClassifier(activation='logistic',
                                   hidden_layer_sizes=(2,))],
            "MLPClassifierLogistic", raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.5.0"),
        reason="ArgMax is missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_mlpclassifier_identity_64(self):
        self._common_classifier(
            [lambda: MLPClassifier(activation='identity',
                                   hidden_layer_sizes=(2,))],
            "MLPClassifierIdentity", raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, TopK are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_knn_64(self):
        self._common_classifier(
            [lambda: KNeighborsClassifier()],
            "KNeighborsClassifier", raw_scores=False)

    @unittest.skipIf(
        VotingClassifier is None, reason="scikit-learn too old")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Sum are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_voting_64(self):
        estimators = [('a', LogisticRegression()),
                      ('b', LogisticRegression())]
        self._common_classifier(
            [lambda: VotingClassifier(estimators,
                                      flatten_transform=False)],
            "VotingClassifier", raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, LpNormalization are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_ovr_64(self):
        self._common_classifier(
            [lambda: OneVsRestClassifier(LogisticRegression())],
            "VotingClassifier", raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, LpNormalization are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_svc_linear_64(self):
        self._common_classifier(
            [lambda: SVC(kernel='linear')], "SVCLinear",
            raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Sum are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_svc_poly_64(self):
        self._common_classifier(
            [lambda: SVC(kernel='poly')], "SVCpoly",
            raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Sum are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_svc_rbf_64(self):
        self._common_classifier(
            [lambda: SVC(kernel='rbf')], "SVCrbf",
            raw_scores=False)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Sum are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_svc_sigmoid_64(self):
        self._common_classifier(
            [lambda: SVC(kernel='sigmoid')], "SVCsigmoid",
            raw_scores=False)

    @unittest.skipIf(
        BernoulliNB is None, reason="new in scikit version 0.20")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Log are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_bernoullinb_64(self):
        self._common_classifier(
            [lambda: BernoulliNB()], "BernoulliNB", raw_scores=False)

    @unittest.skipIf(
        ComplementNB is None, reason="new in scikit version 0.20")
    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, ReduceLogSumExp are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_complementnb_64(self):
        self._common_classifier(
            [lambda: ComplementNB()], "ComplementNB",
            raw_scores=False, pos_features=True)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, ReduceMean are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_bagging_64(self):
        self._common_classifier(
            [lambda: BaggingClassifier(LogisticRegression())],
            "BaggingClassifier")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="ArgMax, Sigmoid are missing")
    @ignore_warnings(category=warnings_to_skip)
    def test_stacking_64(self):
        self._common_classifier(
            [lambda: StackingClassifier([
                ('a', LogisticRegression()),
                ('b', LogisticRegression())])],
            "StackingClassifier")


if __name__ == "__main__":
    unittest.main()
