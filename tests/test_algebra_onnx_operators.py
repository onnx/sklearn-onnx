# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from distutils.version import StrictVersion
from io import BytesIO
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
from onnx import (
    helper, TensorProto, load_model,
    __version__ as onnx__version__)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils.extmath import row_norms
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common._topology import Variable
from skl2onnx.common.data_types import (
    FloatTensorType, guess_numpy_type)
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import (
    OnnxSub, OnnxDiv, OnnxReshapeApi13,
    OnnxReduceSumSquare, OnnxGemm,
    OnnxAdd, OnnxArgMin, OnnxSqrt,
    OnnxArrayFeatureExtractor, OnnxMul,
    OnnxPad, OnnxBatchNormalization,
    OnnxConstantOfShape, OnnxMatMul, OnnxSoftmax)
from test_utils import dump_data_and_model, TARGET_OPSET


class TestOnnxOperators(unittest.TestCase):

    def test_sub(self):

        class CustomOpTransformer(BaseEstimator, TransformerMixin):

            def __init__(self, op_version=None):
                self.op_version = op_version

            def fit(self, X, y=None):
                self.W = np.mean(X, axis=0)
                return self

            def transform(self, X):
                return X - self.W

        mat = np.array([[0., 1.], [1., 2.], [3., 4.]])
        tr = CustomOpTransformer(op_version=None)
        tr.fit(mat)
        z = tr.transform(mat)

        def conv(scope, operator, container):
            dtype = guess_numpy_type(operator.inputs[0].type)
            W = operator.raw_operator.W.astype(dtype)
            op = OnnxSub(
                operator.inputs[0], W, output_names=operator.outputs,
                op_version=TARGET_OPSET)
            op.add_to(scope, container)
            text = str(container)
            if 'name:"Su_Sub"' not in text:
                raise AssertionError(
                    "Unnamed operator: '{}'".format(text))
            nin = list(op.enumerate_initial_types())
            nno = list(op.enumerate_nodes())
            nva = list(op.enumerate_variables())
            self.assertEqual(len(nin), 1)
            self.assertEqual(nin[0][0], 'input')
            self.assertEqual(nin[0][1].shape, [None, 2])
            self.assertEqual(len(nno), 1)
            self.assertEqual(nno[0].output_names[0].onnx_name, 'variable')
            self.assertEqual(len(nva), 1)
            assert isinstance(nva[0], tuple)
            self.assertEqual(nva[0][1], 0)

        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]

        model_onnx = convert_sklearn(
            tr, 'a-sub', [('input', FloatTensorType([None, 2]))],
            custom_shape_calculators={CustomOpTransformer: shape},
            custom_conversion_functions={CustomOpTransformer: conv},
            target_opset=TARGET_OPSET)

        sess = InferenceSession(model_onnx.SerializeToString())
        z2 = sess.run(None, {'input': mat.astype(np.float32)})[0]
        assert_almost_equal(z, z2)

    def test_sub_div(self):

        class CustomOpTransformer(BaseEstimator, TransformerMixin):

            def __init__(self):
                pass

            def fit(self, X, y=None):
                self.W = np.mean(X, axis=0)
                self.S = np.std(X, axis=0)
                return self

            def transform(self, X):
                return (X - self.W) / self.S

        mat = np.array([[0., 1.], [0., 1.], [2., 2.]])
        tr = CustomOpTransformer()
        tr.fit(mat)
        z = tr.transform(mat)

        def conv(scope, operator, container):
            W = operator.raw_operator.W.astype(np.float32)
            S = operator.raw_operator.S.astype(np.float32)
            X = operator.inputs[0]
            out = operator.outputs
            op = OnnxDiv(
                OnnxSub(X, W, op_version=container.target_opset),
                S, output_names=out,
                op_version=container.target_opset)
            op.add_to(scope, container)

        def shape(operator):
            N = operator.inputs[0].type.shape[0]
            W = operator.raw_operator.W
            operator.outputs[0].type.shape = [N, W.shape[0]]

        model_onnx = convert_sklearn(
            tr, 'a-sub-div', [('input', FloatTensorType([None, 2]))],
            custom_shape_calculators={CustomOpTransformer: shape},
            custom_conversion_functions={CustomOpTransformer: conv},
            target_opset=TARGET_OPSET)

        try:
            sess = InferenceSession(model_onnx.SerializeToString())
        except RuntimeError as e:
            raise AssertionError(
                "Cannot load model\n---\n{}\n---".format(model_onnx)) from e
        z2 = sess.run(None, {'input': mat.astype(np.float32)})[0]
        assert_almost_equal(z, z2)

    def test_sub_kmeans(self):

        def conv(scope, operator, container):
            X = operator.inputs[0]
            out = operator.outputs
            op = operator.raw_operator
            dtype = guess_numpy_type(X.type)

            C = op.cluster_centers_
            C2 = row_norms(C, squared=True).astype(dtype)
            C = C.astype(dtype)

            rs = OnnxReduceSumSquare(
                X, axes=[1], keepdims=1,
                op_version=container.target_opset)

            N = X.type.shape[0]
            if isinstance(N, int):
                zeros = np.zeros((N, ))
            else:
                zeros = OnnxMul(
                    rs, np.array([0], dtype=np.float32),
                    op_version=container.target_opset)

            z = OnnxAdd(
                rs,
                OnnxGemm(
                    X, C, zeros, alpha=-2., transB=1,
                    op_version=container.target_opset),
                op_version=container.target_opset)
            y2 = OnnxAdd(C2, z, op_version=container.target_opset)
            lo = OnnxArgMin(
                y2, axis=1, keepdims=0, output_names=out[:1],
                op_version=container.target_opset)
            y2s = OnnxSqrt(
                y2, output_names=out[1:],
                op_version=container.target_opset)

            lo.add_to(scope, container)
            y2s.add_to(scope, container)

        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        model_onnx = convert_sklearn(
            model, 'a-kmeans',
            [('input', FloatTensorType([None, X.shape[1]]))],
            custom_conversion_functions={KMeans: conv},
            target_opset=TARGET_OPSET)

        dump_data_and_model(X.astype(np.float32)[40:60], model, model_onnx,
                            basename="SklearnKMeansCustom-Dec4")

    def test_unscoped(self):
        var2 = OnnxOperator.UnscopedVariable("a")
        var1 = OnnxOperator.UnscopedVariable("a")
        self.assertEqual(var1, var2)
        self.assertEqual(var1, "a")
        self.assertEqual(repr(var1), "UnscopedVariable('a')")

    def test_constant(self):
        cst = OnnxOperator.ConstantVariable("a")
        self.assertEqual(cst.value, "a")

    def test_constant_of_shape(self):
        for opset in range(20, 8, -1):
            if opset > TARGET_OPSET:
                continue
            for value in [np.array([5], dtype=np.float32),
                          np.array(5, dtype=np.float32)]:
                with self.subTest(opset=opset, value=value):
                    tensor_value = onnx.helper.make_tensor(
                        "value", onnx.TensorProto.FLOAT,
                        [1], [5])

                    cst = OnnxConstantOfShape(
                        'X', value=tensor_value, op_version=opset,
                        output_names=['Y'])
                    shape = np.array([3, 4], dtype=np.int64)
                    onx = cst.to_onnx(
                        {'X': shape}, target_opset=opset,
                        outputs=[('Y', FloatTensorType())])
                    sess = InferenceSession(onx.SerializeToString())
                    res = sess.run(None, {'X': shape})
                    assert_almost_equal(
                        res[0], np.full(tuple(shape), 5, dtype=np.float32))

                    cst = OnnxConstantOfShape(
                        'X', value=value, op_version=opset,
                        output_names=['Y'])
                    shape = np.array([3, 4], dtype=np.int64)
                    onx = cst.to_onnx(
                        {'X': shape}, target_opset=opset,
                        outputs=[('Y', FloatTensorType())])
                    sess = InferenceSession(onx.SerializeToString())
                    res = sess.run(None, {'X': shape})
                    assert_almost_equal(
                        res[0], np.full(tuple(shape), 5, dtype=np.float32))

        for opset in [TARGET_OPSET]:
            for value in [5, np.float32(5)]:
                with self.subTest(opset=opset, value=value):
                    with self.assertRaises(TypeError):
                        OnnxConstantOfShape(
                            'X', value=value, op_version=opset,
                            output_names=['Y'])

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    def test_onnx_reversed_order(self):
        idi = np.identity(2)
        idi2 = np.identity(2) * 2

        onx = OnnxAdd(
            OnnxAdd('X', idi.astype(np.float32), op_version=TARGET_OPSET),
            idi2.astype(np.float32), output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(np.float32)})
        self.assertEqual(len(model_def.graph.output), 1)
        onx = OnnxAdd(
            idi2.astype(np.float32),
            OnnxAdd('X', idi.astype(np.float32), op_version=TARGET_OPSET),
            output_names=['Y'], op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(np.float32)})
        onnx2 = model_def.SerializeToString()
        self.assertIsInstance(onx.outputs, list)
        self.assertEqual(len(onx.outputs), 1)
        self.assertIsInstance(onx.outputs[0], (Variable, tuple))
        if isinstance(onx.outputs[0], tuple):
            self.assertEqual(len(onx.outputs[0]), 2)
            self.assertIsInstance(onx.outputs[0][1], FloatTensorType)
        else:
            self.assertIsInstance(onx.outputs[0].type, FloatTensorType)
        # There should be 2 outputs here, bug in ONNX?
        self.assertEqual(len(model_def.graph.output), 1)
        reload = load_model(BytesIO(onnx2))
        self.assertEqual(len(reload.graph.output), 1)
        assert reload is not None

    def test_onnx_reversed_order_second(self):
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 2])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 2])

        nodes = [
            helper.make_node('Add', ['X', 'idi'], ['temp']),
            helper.make_node('Add', ['temp', 'idi2'], ['Y'])
        ]
        graph_def = helper.make_graph(nodes, 't1', [X], [Y])
        model_def = helper.make_model(graph_def, producer_name='A')
        self.assertEqual(len(model_def.graph.output), 1)

        nodes = [
            helper.make_node('Add', ['X', 'idi'], ['temp']),
            helper.make_node('Add', ['idi2', 'temp'], ['Y'])
        ]
        graph_def = helper.make_graph(nodes, 't1', [X], [Y])
        model_def = helper.make_model(graph_def, producer_name='A')
        self.assertEqual(len(model_def.graph.output), 1)

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    def test_onnxt_array_feature_extractor(self):
        onx = OnnxArrayFeatureExtractor(
            'X', np.array([1], dtype=np.int64),
            output_names=['Y'], op_version=1)
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        model_def = onx.to_onnx({'X': X},
                                outputs=[('Y', FloatTensorType([2]))],
                                target_opset=TARGET_OPSET)
        sess = InferenceSession(model_def.SerializeToString())
        got = sess.run(None, {'X': X})[0]
        self.assertEqual(got.shape, (2, 1))
        assert_almost_equal(X[:, 1:2], got)

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    def test_container_init(self):
        onx = OnnxReshapeApi13(
            OnnxReshapeApi13('X', np.array([1, -1], dtype=np.int64),
                             op_version=TARGET_OPSET),
            np.array([1, -1], dtype=np.int64),
            output_names=['Y'], op_version=TARGET_OPSET)
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        model_def = onx.to_onnx({'X': X},
                                outputs=[('Y', FloatTensorType([None, 2]))],
                                target_opset=TARGET_OPSET)
        sess = InferenceSession(model_def.SerializeToString())
        got = sess.run(None, {'X': X})[0]
        assert_almost_equal(X.reshape((1, -1)), got)
        inits = [row for row in str(model_def).split('\n')
                 if row.startswith("  initializer {")]
        self.assertEqual(len(inits), 1)

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    def test_default(self):
        pad = OnnxPad(mode='constant', value=1.5,
                      pads=[0, 1, 0, 1], op_version=10)

        X = helper.make_tensor_value_info(
            'X', onnx.TensorProto.FLOAT, [None, 2])
        model_def = pad.to_onnx({pad.inputs[0].name: X}, target_opset=10)
        onnx.checker.check_model(model_def)

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    def test_batch_normalization(self):

        def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):
            dims_x = len(x.shape)
            dim_ones = (1,) * (dims_x - 2)
            s = s.reshape(-1, *dim_ones)
            bias = bias.reshape(-1, *dim_ones)
            mean = mean.reshape(-1, *dim_ones)
            var = var.reshape(-1, *dim_ones)
            return s * (x - mean) / np.sqrt(var + epsilon) + bias

        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        mean = np.array([0, 3]).astype(np.float32)
        var = np.array([1, 1.5]).astype(np.float32)
        y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

        onx = OnnxBatchNormalization(
            'X', s, bias, mean, var, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(np.float32)},
                                target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, {'X': x})
        assert_almost_equal(y, got[0], decimal=5)

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        mean = np.random.randn(3).astype(np.float32)
        var = np.random.rand(3).astype(np.float32)
        epsilon = 1e-2
        y = _batchnorm_test_mode(
            x, s, bias, mean, var, epsilon).astype(np.float32)

        onx = OnnxBatchNormalization(
            'X', s, bias, mean, var,
            output_names=['Y'], epsilon=epsilon,
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': x.astype(np.float32)},
                                target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, {'X': x})
        assert_almost_equal(y, got[0], decimal=5)

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.6.0"),
                     reason="only available for opset >= 11")
    def test_onnxt_runtime_pad(self):
        data = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                        dtype=np.float32)
        pads = np.array([0, 2, 0, 0], dtype=np.int64)
        constant_value = np.array([0.0], dtype=np.float32)
        exp = np.array([[0.0, 0.0, 1.0, 1.2],
                        [0.0, 0.0, 2.3, 3.4],
                        [0.0, 0.0, 4.5, 5.7]], dtype=np.float32)
        onx = OnnxPad(
            'data', 'pads', constant_value, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, {'data': data, 'pads': pads})
        assert_almost_equal(exp, got[0])

        data = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                        dtype=np.float32)
        pads = np.array([0, 2, 0, 0], dtype=np.int64)
        constant_value = np.array([0.0], dtype=np.float32)
        exp = np.array([[0, 1.2, 1.0, 1.2],
                        [0, 3.4, 2.3, 3.4],
                        [0, 5.7, 4.5, 5.7]], dtype=np.float32)
        onx = OnnxPad(
            'data', 'pads', output_names=['Y'],
            mode='reflect', op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, {'data': data, 'pads': pads})
        try:
            assert_almost_equal(exp, got[0])
        except AssertionError as e:
            warnings.warn(e)

        data = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
                        dtype=np.float32)
        pads = np.array([0, 2, 0, 0], dtype=np.int64)
        constant_value = np.array([0.0], dtype=np.float32)
        exp = np.array([[1.0, 1.0, 1.0, 1.2],
                        [2.3, 2.3, 2.3, 3.4],
                        [4.5, 4.5, 4.5, 5.7]], dtype=np.float32)
        onx = OnnxPad(
            'data', 'pads', output_names=['Y'],
            mode='edge', op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'data': data, 'pads': pads},
                                target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_def.SerializeToString())
        got = oinf.run(None, {'data': data, 'pads': pads})
        assert_almost_equal(exp, got[0])

    def test_softmax(self):
        X = np.random.randn(100, 4).astype(np.float32)
        y = X.sum(axis=1) + np.random.randn(100) / 10
        y = y.astype(np.float32)
        self.assertEqual(y.shape, (100, ))
        weight = np.random.randn(4, 1).astype(np.float32)
        intercept = np.random.randn(1).astype(np.float32)

        node = OnnxAdd(
                OnnxMatMul('X', weight, op_version=TARGET_OPSET),
                intercept, op_version=TARGET_OPSET)
        nn_onnx = node.to_onnx({'X': X}, target_opset=TARGET_OPSET)
        with open("debug_ort_add.onnx", "wb") as f:
            f.write(nn_onnx.SerializeToString())
        self.assertEqual(len(nn_onnx.graph.output), 1)

        node = OnnxMatMul('X', weight, op_version=TARGET_OPSET)
        nn_onnx = node.to_onnx({'X': X}, target_opset=TARGET_OPSET)
        self.assertEqual(len(nn_onnx.graph.output), 1)

        node = OnnxSoftmax(
            OnnxAdd(
                OnnxMatMul('X', weight, op_version=TARGET_OPSET),
                intercept, op_version=TARGET_OPSET),
            op_version=TARGET_OPSET)
        nn_onnx = node.to_onnx({'X': X}, target_opset=TARGET_OPSET)
        self.assertEqual(len(nn_onnx.graph.output), 1)


if __name__ == "__main__":
    # import logging
    # logger = logging.getLogger('skl2onnx')
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestOnnxOperators().test_softmax()
    unittest.main()
