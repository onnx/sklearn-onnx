# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from test_utils import dump_one_class_classification, dump_binary_classification, dump_multiple_classification
from test_utils import dump_multiple_regression, dump_single_regression


class TestSklearnTreeEnsembleModels(unittest.TestCase):

    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        dump_one_class_classification(model,
                                      allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                                                    "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")
        dump_binary_classification(model,
                                   allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                                                 "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")
        dump_multiple_classification(model,
                                     allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                                                   "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=3)
        dump_single_regression(model, allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")
        dump_multiple_regression(model, allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_extra_trees_classifier(self):
        model = ExtraTreesClassifier(n_estimators=3)
        dump_one_class_classification(model,
                                      allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                                                    "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")
        dump_binary_classification(model,
                                   allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                                                 "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")
        dump_multiple_classification(model,
                                     # Operator cast-1 is not implemented in onnxruntime
                                     allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2') or "
                                                   "StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")

    def test_extra_trees_regressor(self):
        model = ExtraTreesRegressor(n_estimators=3)
        dump_single_regression(model, allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")
        dump_multiple_regression(model, allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.2.1')")




if __name__ == "__main__":
    unittest.main()
