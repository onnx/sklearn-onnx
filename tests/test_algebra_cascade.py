import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnx.defs import onnx_opset_version
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph, Fail
except ImportError:
    InvalidGraph = RuntimeError
    Fail = RuntimeError
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxScaler
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn
from test_utils import fit_regression_model


class TestOnnxOperatorsCascade(unittest.TestCase):

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_cascade_add(self):

        def generate_onnx_graph(dim, nbnode, input_name='X1', opv=None):
            i1 = input_name
            for i in range(nbnode - 1):
                i2 = (np.ones((1, dim)) * nbnode * 10).astype(np.float32)
                node = OnnxAdd(i1, i2, op_version=opv)
                i1 = node
            i2 = (np.ones((1, dim)) * nbnode * 10).astype(np.float32)
            node = OnnxAdd(i1, i2, output_names=['Y'], op_version=opv)
            onx = node.to_onnx([(input_name, FloatTensorType((None, dim)))],
                               outputs=[('Y', FloatTensorType())],
                               target_opset=opv)
            return onx

        exp = [np.array([[11., 11., 11., 11., 11.]]),
               np.array([[42., 42., 42., 42., 42.]]),
               np.array([[93., 93., 93., 93., 93.]]),
               np.array([[100100., 100100., 100100., 100100., 100100.]])]
        for opv in ({'': 10}, None, 9, 10, 11, onnx_opset_version()):
            for i, nbnode in enumerate((1, 2, 3, 100)):
                onx = generate_onnx_graph(5, nbnode, opv=opv)
                as_string = onx.SerializeToString()
                try:
                    ort = InferenceSession(as_string)
                except InvalidGraph as e:
                    if opv >= onnx_opset_version():
                        continue
                    raise AssertionError(
                        "Unable to load opv={}\n---\n{}\n---".format(
                            opv, onx)) from e
                X = (np.ones((1, 5)) * nbnode).astype(np.float32)
                res_out = ort.run(None, {'X1': X})
                assert len(res_out) == 1
                res = res_out[0]
                assert_almost_equal(exp[i], res)

        dim = 10
        onx = generate_onnx_graph(dim, 300, opv=11)
        as_string = onx.SerializeToString()
        ort = InferenceSession(as_string)
        X = (np.ones((1, dim)) * nbnode).astype(np.float32)
        res_out = ort.run(None, {'X1': X})
        assert len(res_out) == 1
        res = res_out[0]
        assert res.shape[1] == dim

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion("1.4.0"),
                     reason="not available")
    def test_cascade_scaler(self):

        def generate_onnx_graph(dim, nbnode, input_name='X1', opv=1):
            i1 = input_name
            scale = list(np.ones((1, dim)).ravel())
            for i in range(nbnode - 1):
                i2 = list(map(float, np.ones((1, dim)).astype(
                    np.float32).ravel()))
                node = OnnxScaler(i1, offset=i2, scale=scale, op_version=opv)
                i1 = node
            i2 = list(map(float, np.ones((1, dim)).astype(np.float32).ravel()))
            node = OnnxScaler(i1, offset=i2, scale=scale, output_names=['Y'],
                              op_version=opv)
            onx = node.to_onnx([(input_name, FloatTensorType((None, dim)))],
                               outputs=[('Y', FloatTensorType((None, dim)))])
            return onx

        exp = [np.zeros((1, 5)),
               np.zeros((1, 5)),
               np.zeros((1, 5)),
               np.zeros((1, 5))]
        for opv in (1, 2, 3, None):
            for i, nbnode in enumerate((1, 2, 3, 100)):
                onx = generate_onnx_graph(5, nbnode, opv=opv)
                as_string = onx.SerializeToString()
                try:
                    ort = InferenceSession(as_string)
                except InvalidGraph as e:
                    if opv in (3, ):
                        continue
                    if opv >= onnx_opset_version():
                        continue
                    raise AssertionError(
                        "Unable to load opv={}\n---\n{}\n---".format(
                            opv, onx)) from e
                X = (np.ones((1, 5)) * nbnode).astype(np.float32)
                res_out = ort.run(None, {'X1': X})
                assert len(res_out) == 1
                res = res_out[0]
                assert_almost_equal(exp[i], res)

        dim = 10
        onx = generate_onnx_graph(dim, 300)
        as_string = onx.SerializeToString()
        ort = InferenceSession(as_string)
        X = (np.ones((1, dim)) * nbnode).astype(np.float32)
        res_out = ort.run(None, {'X1': X})
        assert len(res_out) == 1
        res = res_out[0]
        assert res.shape[1] == dim

    def test_scaler_converted(self):
        st = StandardScaler()
        X = np.array([[0, 1.5], [6.1, 2.3]])
        st.fit(X)
        exp = st.transform(X)

        for opv in (1, 2, 10, 11, None, onnx_opset_version()):
            try:
                onx = to_onnx(st, X.astype(np.float32), target_opset=opv)
            except RuntimeError as e:
                if ("is higher than the number of the "
                        "installed onnx package") in str(e):
                    continue
                raise e
            as_string = onx.SerializeToString()
            try:
                ort = InferenceSession(as_string)
            except InvalidGraph as e:
                if opv > onnx_opset_version():
                    continue
                raise AssertionError(
                    "Unable to load opv={}\n---\n{}\n---".format(
                        opv, onx)) from e
            res_out = ort.run(None, {'X': X.astype(np.float32)})
            assert len(res_out) == 1
            res = res_out[0]
            assert_almost_equal(exp, res)

        for opv in (1, 2, 10, 11, None, onnx_opset_version()):
            onx = to_onnx(st, X.astype(np.float32),
                          target_opset={'ai.onnx.ml': opv})
            as_string = onx.SerializeToString()
            try:
                ort = InferenceSession(as_string)
            except InvalidGraph as e:
                if opv > onnx_opset_version():
                    continue
                raise AssertionError(
                    "Unable to load opv={}\n---\n{}\n---".format(
                        opv, onx)) from e
            res_out = ort.run(None, {'X': X.astype(np.float32)})
            assert len(res_out) == 1
            res = res_out[0]
            assert_almost_equal(exp, res)

    def test_model_mlp_regressor_default(self):
        model, X_test = fit_regression_model(
            MLPRegressor(random_state=42))
        exp = model.predict(X_test)
        for opv in (1, 2, 7, 8, 9, 10, 11, None, onnx_opset_version()):
            try:
                onx = convert_sklearn(
                    model, "scikit-learn MLPRegressor",
                    [("input", FloatTensorType([None, X_test.shape[1]]))],
                    target_opset=opv)
            except RuntimeError as e:
                if ("is higher than the number of the "
                        "installed onnx package") in str(e):
                    continue
                raise e
            as_string = onx.SerializeToString()
            try:
                ort = InferenceSession(as_string)
            except (RuntimeError, InvalidGraph, Fail) as e:
                if opv in (1, 2):
                    continue
                if opv >= onnx_opset_version():
                    continue
                raise AssertionError(
                    "Unable to load opv={}\n---\n{}\n---".format(
                        opv, onx)) from e
            res_out = ort.run(None, {'input': X_test})
            assert len(res_out) == 1
            res = res_out[0]
            assert_almost_equal(exp.ravel(), res.ravel(), decimal=4)


if __name__ == "__main__":
    unittest.main()
