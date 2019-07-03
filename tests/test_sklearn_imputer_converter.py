"""
Tests scikit-imputer converter.
"""
import unittest
import numpy as np
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
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from test_utils import dump_data_and_model


class TestSklearnImputerConverter(unittest.TestCase):
    @unittest.skipIf(Imputer is None,
                     reason="Imputer removed in 0.21")
    def test_model_imputer(self):
        model = Imputer(missing_values="NaN", strategy="mean", axis=0)
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)
        # The conversion works but internally scikit-learn converts
        # everything into float before looking into missing values.
        # There is no nan integer. The runtime is not tested
        # in this case.
        model_onnx = convert_sklearn(model, "scikit-learn imputer",
                                     [("input", Int64TensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)

    @unittest.skipIf(Imputer is None,
                     reason="Imputer removed in 0.21")
    def test_imputer_int_inputs(self):
        model = Imputer(missing_values="NaN", strategy="mean", axis=0)
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scikit-learn imputer",
                                     [("input", Int64TensorType([1, 2]))])
        self.assertEqual(len(model_onnx.graph.node), 2)

        # Last node should be Imputer
        outputs = model_onnx.graph.output
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].type.tensor_type.shape.dim[-1].dim_value,
                         2)

    @unittest.skipIf(Imputer is None,
                     reason="Imputer removed in 0.21")
    def test_imputer_float_inputs(self):
        model = Imputer(missing_values="NaN", strategy="mean", axis=0)
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)

        model_onnx = convert_sklearn(model, "scikit-learn imputer",
                                     [("input", FloatTensorType([1, 2]))])
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
            [("input", FloatTensorType([1, 2]))],
        )
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
            basename="SklearnSimpleImputerMeanFloat32",
        )


if __name__ == "__main__":
    unittest.main()
