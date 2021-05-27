# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's cast transformer converter.
"""
import unittest
import numpy
from sklearn.pipeline import Pipeline
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
from skl2onnx.sklapi import ReplaceTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnCastTransformerConverter(unittest.TestCase):

    def common_test_replace_transformer(self, dtype, input_type):
        model = Pipeline([
            ('replace', ReplaceTransformer(dtype=numpy.float32)),
        ])
        data = numpy.array([[0.1, 0.2, 3.1], [1, 1, 0],
                            [0, 2, 1], [1, 0, 2]],
                           dtype=numpy.float32)
        model.fit(data)
        pred = model.steps[0][1].transform(data)
        assert pred.dtype == dtype
        model_onnx = convert_sklearn(
            model, "cast", [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnCastTransformer{}".format(
                input_type.__class__.__name__))

    @unittest.skipIf(TARGET_OPSET < 11, reason="not supported")
    def test_replace_transformer(self):
        self.common_test_replace_transformer(
            numpy.float32, FloatTensorType)


if __name__ == "__main__":
    unittest.main()
