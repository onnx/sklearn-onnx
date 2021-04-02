# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
import numpy
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType, FloatTensorType
from test_utils import dump_data_and_model


class TestSklearnScalerConverter(unittest.TestCase):
    def test_standard_scaler(self):
        model = StandardScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", Int64TensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.int64),
            model,
            model_onnx,
            basename="SklearnStandardScalerInt64",
        )

    def test_standard_scaler_floats(self):
        model = StandardScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnStandardScalerFloat32",
        )

    def test_standard_scaler_floats_div(self):
        model = StandardScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "scaler", [("input", FloatTensorType([None, 3]))],
            options={id(model): {'div': 'div'}})
        assert 'op_type: "Div"' in str(model_onnx)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32Div")

    def test_standard_scaler_floats_div_cast(self):
        model = StandardScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(
            model, "cast", [("input", FloatTensorType([None, 3]))],
            options={id(model): {'div': 'div_cast'}})
        assert 'op_type: "Div"' in str(model_onnx)
        assert 'caler"' not in str(model_onnx)
        assert "double_data:" in str(model_onnx)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32DivCast")

    def test_standard_scaler_floats_no_std(self):
        model = StandardScaler(with_std=False)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnStandardScalerFloat32NoStd",
        )

    def test_standard_scaler_floats_no_mean(self):
        model = StandardScaler(with_mean=False)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnStandardScalerFloat32NoMean",
        )

    def test_standard_scaler_floats_no_mean_std(self):
        model = StandardScaler(with_mean=False, with_std=False)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnStandardScalerFloat32NoMeanStd",
        )

    def test_robust_scaler_floats(self):
        model = RobustScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnRobustScalerFloat32",
        )

    def test_robust_scaler_floats_no_bias(self):
        model = RobustScaler(with_centering=False)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnRobustScalerWithCenteringFloat32",
        )

    def test_robust_scaler_floats_no_scaling(self):
        model = RobustScaler(with_scaling=False)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnRobustScalerNoScalingFloat32",
        )

    def test_robust_scaler_floats_no_centering_scaling(self):
        model = RobustScaler(with_centering=False, with_scaling=False)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnRobustScalerNoCenteringScalingFloat32",
        )

    def test_min_max_scaler(self):
        model = MinMaxScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnMinMaxScaler",
        )

    def test_max_abs_scaler(self):
        model = MaxAbsScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnMaxAbsScaler")


if __name__ == "__main__":
    unittest.main()
