# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion
import numpy as np
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from skl2onnx.common.data_types import onnx_built_with_ml
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx import convert_sklearn
from test_utils import (
    dump_one_class_classification,
    dump_binary_classification,
    dump_multiple_classification,
)
from test_utils import dump_data_and_model, fit_regression_model
from test_utils import dump_multiple_regression, dump_single_regression
from onnxruntime import InferenceSession, __version__


class TestSklearnDecisionTreeModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(__version__) <= StrictVersion("0.3.0"),
        reason="No suitable kernel definition found "
               "for op Cast(9) (node Cast)")
    def test_decisiontree_classifier1(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict_proba(X)
        if res[1][0][0] != pred[0, 0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decisiontree_regressor0(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        if res[0][0, 0] != pred[0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        dump_one_class_classification(
            model,
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_binary_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    def test_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        dump_single_regression(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )
        dump_multiple_regression(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def test_decision_tree_regressor_int(self):
        model, X = fit_regression_model(
            DecisionTreeRegressor(random_state=42), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "decision tree regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnDecisionTreeRegressionInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )


if __name__ == "__main__":
    unittest.main()
