# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-imputer converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
import sklearn
try:
    from sklearn.preprocessing import Imputer
except ImportError:
    Imputer = None

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    # changed in 0.20
    SimpleImputer = None

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, StringTensorType)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnImputerConverter(unittest.TestCase):

    def _check_outputs_ints(self, model, model_onnx, data):
        sess = InferenceSession(model_onnx.SerializeToString())
        idata = {'input': np.array(data).astype(np.int64)}
        res = sess.run(None, idata)[0]
        exp = model.transform(data)
        assert_almost_equal(res, exp)

    def _check_outputs_strings(self, model, model_onnx, data):
        idata = {'input': np.array(data).astype(np.str_)}
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, idata)[0]
        exp = model.transform(data)
        if list(exp.ravel()) != list(res.ravel()):
            raise AssertionError(
                "Unexpected output expected %r != %r." % (exp, res))

    @unittest.skipIf(Imputer is None,
                     reason="Imputer removed in 0.21")
    def test_imputer_float_inputs(self):
        model = Imputer(missing_values="NaN", strategy="mean", axis=0)
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)

        model_onnx = convert_sklearn(model, "scikit-learn imputer",
                                     [("input", FloatTensorType([None, 2]))])
        self.assertTrue(model_onnx.graph.node is not None)

        # should contain only node
        self.assertEqual(len(model_onnx.graph.node), 1)

        # last node should contain the Imputer
        outputs = model_onnx.graph.output
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].type.tensor_type.shape.dim[-1].dim_value,
                         2)
        dump_data_and_model(
            np.array(data, dtype=np.float32),
            model,
            model_onnx,
            basename="SklearnImputerMeanFloat32",
        )

    @unittest.skipIf(SimpleImputer is None,
                     reason="SimpleImputer changed in 0.20")
    def test_simple_imputer_float_inputs(self):
        model = SimpleImputer(strategy="mean", fill_value="nan")
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn simple imputer",
            [("input", FloatTensorType([None, 2]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx.graph.node is not None)

        # should contain only node
        self.assertEqual(len(model_onnx.graph.node), 1)

        # last node should contain the Imputer
        outputs = model_onnx.graph.output
        self.assertEqual(len(outputs), 1)
        self.assertEqual(
            outputs[0].type.tensor_type.shape.dim[-1].dim_value, 2)
        dump_data_and_model(
            np.array(data, dtype=np.float32),
            model, model_onnx,
            basename="SklearnSimpleImputerMeanFloat32")

    @unittest.skipIf(SimpleImputer is None,
                     reason="SimpleImputer changed in 0.20")
    def test_simple_imputer_float_inputs_int_mostf(self):
        model = SimpleImputer(strategy="most_frequent", fill_value="nan")
        data = [[1, 2], [np.nan, 3], [7, 6], [8, np.nan]]
        model.fit(data)

        model_onnx = convert_sklearn(
            model,
            "scikit-learn simple imputer",
            [("input", Int64TensorType([None, 2]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx.graph.node is not None)

        # should contain only node
        self.assertEqual(len(model_onnx.graph.node), 1)

        # last node should contain the Imputer
        outputs = model_onnx.graph.output
        self.assertEqual(len(outputs), 1)
        self._check_outputs_ints(model, model_onnx, data)

    @unittest.skipIf(SimpleImputer is None,
                     reason="SimpleImputer changed in 0.20")
    def test_simple_imputer_float_inputs_int_mean(self):
        model = SimpleImputer(strategy="mean", fill_value="nan")
        data = [[1, 2], [np.nan, 3], [7, 6], [8, np.nan]]
        model.fit(data)

        try:
            convert_sklearn(
                model,
                "scikit-learn simple imputer",
                [("input", Int64TensorType([None, 2]))],
                target_opset=TARGET_OPSET)
        except RuntimeError as e:
            assert "nan values are replaced by float" in str(e)

    @unittest.skipIf(SimpleImputer is None,
                     reason="SimpleImputer changed in 0.20")
    @unittest.skipIf(
        StrictVersion(sklearn.__version__) < StrictVersion('0.24'),
        reason="SimpleImputer does not support strings")
    def test_simple_imputer_string_inputs_int_mostf(self):
        model = SimpleImputer(
            strategy="most_frequent", fill_value="nan", missing_values="")
        data = [["s1", "s2"], ["", "s3"], ["s7", "s6"], ["s8", ""]]
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scikit-learn simple imputer",
            [("input", StringTensorType([None, 2]))],
            target_opset=TARGET_OPSET)
        self.assertIn("ai.onnx.ml", str(model_onnx))
        self.assertTrue(model_onnx.graph.node is not None)
        self.assertEqual(len(model_onnx.graph.output), 1)
        self._check_outputs_strings(model, model_onnx, data)


if __name__ == "__main__":
    unittest.main()
