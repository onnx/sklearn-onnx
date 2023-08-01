# SPDX-License-Identifier: Apache-2.0

import unittest
import numbers
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import update_registered_converter, convert_sklearn
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)  # noqa
from skl2onnx._parse import _parse_sklearn_classifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm,
)  # noqa
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost,
)  # noqa

try:
    from test_utils import fit_classification_model
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_utils import fit_classification_model
from test_utils import TARGET_OPSET, TARGET_OPSET_ML


class TestOptionColumns(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        update_registered_converter(
            LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
        )

        def custom_parser(scope, model, inputs, custom_parsers=None):
            if custom_parsers is not None and model in custom_parsers:
                return custom_parsers[model](
                    scope, model, inputs, custom_parsers=custom_parsers
                )
            if not all(
                isinstance(i, (numbers.Real, bool, np.bool_)) for i in model.classes_
            ):
                raise NotImplementedError(
                    "Current converter does not support string labels."
                )
            return _parse_sklearn_classifier(scope, model, inputs)

        update_registered_converter(
            XGBClassifier,
            "XGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            parser=custom_parser,
            options={"zipmap": [True, False, "columns"], "nocl": [True, False]},
        )

    def c_test_model(self, model):
        model, X = fit_classification_model(model, 3, n_features=4, label_string=False)
        model_onnx = convert_sklearn(
            model,
            "multi-class ridge classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {"zipmap": "columns"}},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self.assertIsNotNone(model_onnx)
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = [_.name for _ in sess.get_outputs()]
        self.assertEqual(["output_label", "i0", "i1", "i2"], names)
        xt = X[:10].astype(np.float32)
        got = sess.run(None, {"input": xt})
        prob = model.predict_proba(xt)
        for i in range(prob.shape[1]):
            assert_almost_equal(prob[:, i], got[i + 1])

    def test_random_forest(self):
        self.c_test_model(RandomForestClassifier(n_estimators=3))

    def test_lightgbm(self):
        self.c_test_model(LGBMClassifier(n_estimators=3))

    def test_xgboost(self):
        self.c_test_model(XGBClassifier(n_estimators=3))


if __name__ == "__main__":
    unittest.main()
