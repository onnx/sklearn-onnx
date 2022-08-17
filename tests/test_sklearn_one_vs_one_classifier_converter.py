# SPDX-License-Identifier: Apache-2.0

import packaging.version as pv
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType)
from test_utils import (
    dump_data_and_model,
    dump_multiple_classification,
    fit_classification_model,
    fit_multilabel_classification_model,
    TARGET_OPSET)

warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


ort_version = '.'.join(ort_version.split('.')[:2])


class TestOneVsOneClassifierConverter(unittest.TestCase):
    def test_one_vs_one_classifier_converter(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
        clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
        result = clf.predict(X_test[:10])
        print(result)

#        result = np.array([2 1 0 2 0 2 0 1 1 1])

if __name__ == "__main__":
    unittest.main()
