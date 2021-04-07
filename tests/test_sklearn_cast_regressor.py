# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's cast transformer converter.
"""
import unittest
import math
from distutils.version import StrictVersion
import numpy
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
from skl2onnx.sklapi import CastRegressor, CastTransformer
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import (
    FloatTensorType, DoubleTensorType)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnCastRegressorConverter(unittest.TestCase):

    def common_test_cast_regressor(self, dtype, input_type):
        model = CastRegressor(DecisionTreeRegressor(max_depth=2), dtype=dtype)
        data = numpy.array([[0.1, 0.2, 3.1], [1, 1, 0],
                            [0, 2, 1], [1, 0, 2],
                            [0.1, 2.1, 1.1], [1.1, 0.1, 2.2],
                            [-0.1, -2.1, -1.1], [-1.1, -0.1, -2.2],
                            [0.2, 2.2, 1.2], [1.2, 0.2, 2.2]],
                           dtype=numpy.float32)
        y = (numpy.sum(data, axis=1, keepdims=0) +
             numpy.random.randn(data.shape[0]))
        model.fit(data, y)
        pred = model
        assert pred.dtype == dtype
        model_onnx = convert_sklearn(
            model, "cast", [("input", FloatTensorType([None, 3]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnCastRegressor{}".format(
                input_type.__class__.__name__))

    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('0.5.0'),
                     reason="runtime too old")
    def test_cast_regressor_float(self):
        self.common_test_cast_regressor(
            numpy.float32, FloatTensorType)

    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('0.5.0'),
                     reason="runtime too old")
    def test_cast_regressor_float64(self):
        self.common_test_cast_regressor(
            numpy.float64, DoubleTensorType)

    @unittest.skipIf(TARGET_OPSET < 9, reason="not supported")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('0.5.0'),
                     reason="runtime too old")
    def test_pipeline(self):

        def maxdiff(a1, a2):
            d = numpy.abs(a1.ravel() - a2.ravel())
            return d.max()

        X, y = make_regression(10000, 10, random_state=3)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=3)
        Xi_train, yi_train = X_train.copy(), y_train.copy()
        Xi_test = X_test.copy()
        for i in range(X.shape[1]):
            Xi_train[:, i] = (Xi_train[:, i] * math.pi * 2 ** i).astype(
                numpy.int64)
            Xi_test[:, i] = (Xi_test[:, i] * math.pi * 2 ** i).astype(
                numpy.int64)
        max_depth = 10
        Xi_test = Xi_test.astype(numpy.float32)

        # model 1
        model1 = Pipeline([
            ('scaler', StandardScaler()),
            ('dt', DecisionTreeRegressor(max_depth=max_depth))
        ])
        model1.fit(Xi_train, yi_train)
        exp1 = model1.predict(Xi_test)
        onx1 = to_onnx(model1, X_train[:1].astype(numpy.float32),
                       target_opset=TARGET_OPSET)
        sess1 = InferenceSession(onx1.SerializeToString())
        got1 = sess1.run(None, {'X': Xi_test})[0]
        md1 = maxdiff(exp1, got1)

        # model 2
        model2 = Pipeline([
            ('cast64', CastTransformer(dtype=numpy.float64)),
            ('scaler', StandardScaler()),
            ('cast', CastTransformer()),
            ('dt', CastRegressor(DecisionTreeRegressor(max_depth=max_depth),
                                 dtype=numpy.float32))
        ])
        model2.fit(Xi_train, yi_train)
        exp2 = model2.predict(Xi_test)
        onx = to_onnx(model2, X_train[:1].astype(numpy.float32),
                      options={StandardScaler: {'div': 'div_cast'}},
                      target_opset=TARGET_OPSET)
        sess2 = InferenceSession(onx.SerializeToString())
        got2 = sess2.run(None, {'X': Xi_test})[0]
        md2 = maxdiff(exp2, got2)
        assert md2 <= md1
        assert md2 <= 0.0


if __name__ == "__main__":
    unittest.main()
