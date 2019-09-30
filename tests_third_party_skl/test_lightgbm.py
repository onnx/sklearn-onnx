# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from lightgbm import LGBMClassifier, LGBMRegressor
from skl2onnx.third_party_skl import register_converters

try:
    from test_utils import dump_single_regression
except ImportError:
    import os
    import sys
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), "..", "tests"))
    from test_utils import dump_single_regression
from test_utils import dump_binary_classification, dump_multiple_classification


class TestLightGbmTreeEnsembleModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        register_converters()

    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        dump_binary_classification(
            model, allow_failure="StrictVersion(onnx.__version__) < "
                                 "StrictVersion('1.3.0')")
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__) < "
                          "StrictVersion('1.3.0')")

    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        dump_single_regression(model)

    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1)
        dump_single_regression(model, suffix="1")

    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        dump_single_regression(model, suffix="2")


if __name__ == "__main__":
    unittest.main()
