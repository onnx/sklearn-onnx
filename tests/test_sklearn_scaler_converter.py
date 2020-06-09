"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
import math
import numpy
from onnxruntime import InferenceSession
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn, to_onnx
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

    def test_standard_scaler_floats_cast(self):
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
            options={id(model): {'double': True}})
        assert 'op_type: "Cast"' in str(model_onnx)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32")

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

    def test_discrepencies(self):
        X, _ = make_regression(100, 10, random_state=3)
        X_train, X_test = train_test_split(X, random_state=3)

        Xi_train = X_train.copy()
        Xi_test = X_test.copy()
        for i in range(X.shape[1]):
            Xi_train[:, i] = (Xi_train[:, i] * math.pi * 2 ** i).astype(
                numpy.int64)
            Xi_test[:, i] = (Xi_test[:, i] * math.pi * 2 ** i).astype(
                numpy.int64)
        for i in range(Xi_test.shape[0]):
            Xi_test[i, :] *= 2 ** i

        Xi_train = Xi_train.astype(numpy.float32)
        model = StandardScaler()
        model.fit(Xi_train)
        X32 = Xi_test.astype(numpy.float32)
        pred = model.transform(X32)

        onx32 = to_onnx(model, Xi_train[:1].astype(numpy.float32))
        sess32 = InferenceSession(onx32.SerializeToString())
        got32 = sess32.run(0, {'X': X32})[0]
        d32 = numpy.max(numpy.abs(pred.ravel() - got32.ravel()))

        onx64 = to_onnx(model, Xi_train[:1].astype(numpy.float32),
                        options={id(model): {'double': True}})
        sess64 = InferenceSession(onx64.SerializeToString())
        got64 = sess64.run(0, {'X': X32})[0]
        d64 = numpy.max(numpy.abs(pred.ravel() - got64.ravel()))

        assert d32 <= d64


if __name__ == "__main__":
    unittest.main()
