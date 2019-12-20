# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numbers
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,  # noqa
    calculate_linear_regressor_output_shapes,
)
from skl2onnx.common.data_types import (
    SequenceType, DictionaryType, Int64TensorType, StringTensorType
)
from onnxmltools.convert.lightgbm._parse import WrappedBooster
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm  # noqa
)

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

        def custom_parser(scope, model, inputs, custom_parsers=None):
            assert WrappedBooster is not None
            if custom_parsers is not None and model in custom_parsers:
                return custom_parsers[model](
                    scope, model, inputs, custom_parsers=custom_parsers)
            if all(isinstance(i, (numbers.Real, bool, np.bool_))
                   for i in model.classes_):
                label_type = Int64TensorType()
            else:
                label_type = StringTensorType()
            output_label = scope.declare_local_variable(
                'output_label', label_type)

            this_operator = scope.declare_local_operator(
                'LgbmClassifier', model)
            this_operator.inputs = inputs
            probability_map_variable = scope.declare_local_variable(
                'output_probability', SequenceType(DictionaryType(
                    label_type, scope.tensor_type())))
            this_operator.outputs.append(output_label)
            this_operator.outputs.append(probability_map_variable)
            return this_operator.outputs

        update_registered_converter(
            LGBMClassifier, 'LgbmClassifier',
            calculate_linear_classifier_output_shapes,
            convert_lightgbm, parser=custom_parser)
        update_registered_converter(
            LGBMRegressor, 'LgbmRegressor',
            calculate_linear_regressor_output_shapes,
            convert_lightgbm)

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
