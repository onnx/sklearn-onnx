# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_one_class_classification,
    dump_binary_classification,
    dump_multiple_classification,
)
from test_utils import dump_multiple_regression, dump_single_regression


class TestSklearnDecisionTreeModels(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
