# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from onnxruntime import __version__ as ort_version
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler)
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    Int64TensorType, FloatTensorType, DoubleTensorType)
from test_utils import dump_data_and_model, TARGET_OPSET


ort_version = ".".join(ort_version.split('.')[:2])


class TestSklearnScalerConverter(unittest.TestCase):

    @ignore_warnings(category=DeprecationWarning)
    def test_standard_scaler_int(self):
        model = StandardScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", Int64TensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.int64),
            model, model_onnx,
            basename="SklearnStandardScalerInt64")

    @ignore_warnings(category=DeprecationWarning)
    def test_min_max_scaler_int(self):
        model = MinMaxScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", Int64TensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.int64),
            model, model_onnx,
            basename="SklearnMinMaxScalerInt64")

    @ignore_warnings(category=DeprecationWarning)
    def test_standard_scaler_double(self):
        model = StandardScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", DoubleTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float64),
            model, model_onnx,
            basename="SklearnStandardScalerDouble")

    @ignore_warnings(category=DeprecationWarning)
    def test_standard_scaler_blacklist(self):
        model = StandardScaler()
        data = numpy.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]],
                           dtype=numpy.float32)
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET,
                                     black_op={'Normalizer', 'Scaler'})
        self.assertNotIn('Normalizer', str(model_onnx))
        self.assertNotIn('Scaler', str(model_onnx))
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnStandardScalerBlackList")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32")

    @ignore_warnings(category=DeprecationWarning)
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

    @ignore_warnings(category=DeprecationWarning)
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
            options={id(model): {'div': 'div_cast'}},
            target_opset=TARGET_OPSET)
        assert 'op_type: "Div"' in str(model_onnx)
        assert 'caler"' not in str(model_onnx)
        assert "double_data:" in str(model_onnx)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32DivCast")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32NoStd")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32NoMean")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnStandardScalerFloat32NoMeanStd")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnRobustScalerFloat32")

    @ignore_warnings(category=DeprecationWarning)
    def test_robust_scaler_doubles(self):
        model = RobustScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", DoubleTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float64),
            model, model_onnx, basename="SklearnRobustScalerFloat64")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnRobustScalerWithCenteringFloat32")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnRobustScalerNoScalingFloat32")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model,
            basename="SklearnRobustScalerNoCenteringScalingFloat32")

    @ignore_warnings(category=DeprecationWarning)
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
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnMinMaxScaler")

    @ignore_warnings(category=DeprecationWarning)
    def test_min_max_scaler_double(self):
        model = MinMaxScaler()
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", DoubleTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float64),
            model, model_onnx, basename="SklearnMinMaxScalerDouble")

    @ignore_warnings(category=DeprecationWarning)
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.9.0"),
                     reason="Operator clip not fully implemented")
    def test_min_max_scaler_clip(self):
        model = MinMaxScaler(clip=True)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        data[0][0] = 1e6
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, model_onnx, basename="SklearnMinMaxScalerClip")

    @ignore_warnings(category=DeprecationWarning)
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.9.0"),
                     reason="Operator clip not fully implemented")
    def test_min_max_scaler_double_clip(self):
        model = MinMaxScaler(clip=True)
        data = [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", DoubleTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        data[0][0] = 1e6
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float64),
            model, model_onnx, basename="SklearnMinMaxScalerDouble")

    @ignore_warnings(category=DeprecationWarning)
    def test_max_abs_scaler(self):
        model = MaxAbsScaler()
        data = [
            [0.0, 0.0, -3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", FloatTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float32),
            model, basename="SklearnMaxAbsScaler")

    @ignore_warnings(category=DeprecationWarning)
    def test_max_abs_scaler_double(self):
        model = MaxAbsScaler()
        data = [
            [0.0, 0.0, -3.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
        model.fit(data)
        model_onnx = convert_sklearn(model, "scaler",
                                     [("input", DoubleTensorType([None, 3]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            numpy.array(data, dtype=numpy.float64),
            model, model_onnx, basename="SklearnMaxAbsScalerDouble")


if __name__ == "__main__":
    unittest.main()
