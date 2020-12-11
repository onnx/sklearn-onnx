import unittest
from distutils.version import StrictVersion
from io import BytesIO
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.utils.extmath import row_norms
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import (
    OnnxSub, OnnxDiv, OnnxReshape,
    OnnxReduceSumSquare, OnnxGemm,
    OnnxAdd, OnnxArgMin, OnnxSqrt,
    OnnxArrayFeatureExtractor, OnnxMul
)
from onnx import (
    helper, TensorProto, load_model,
    __version__ as onnx__version__
)
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
            W = operator.raw_operator.W.astype(container.dtype)
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
            assert len(nin) == 1
            assert nin[0][0] == 'input'
            assert nin[0][1].shape == [None, 2]
            assert len(nno) == 1
            assert nno[0].output_names == ['variable']
            assert len(nva) == 1
            assert isinstance(nva[0], tuple)
            assert nva[0][1] == 0

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

            C = op.cluster_centers_
            C2 = row_norms(C, squared=True).astype(container.dtype)
            C = C.astype(container.dtype)

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

    @unittest.skipIf(StrictVersion(onnx__version__) < StrictVersion("1.4.0"),
                     reason="only available for opset >= 10")
    def test_onnx_reversed_order(self):
        idi = np.identity(2)
        idi2 = np.identity(2) * 2

        onx = OnnxAdd(
            OnnxAdd('X', idi, op_version=TARGET_OPSET),
            idi2, output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(np.float32)})
        self.assertEqual(len(model_def.graph.output), 1)
        onx = OnnxAdd(
            idi2,
            OnnxAdd(
                'X', idi, op_version=TARGET_OPSET),
            output_names=['Y'],
            op_version=TARGET_OPSET)
        model_def = onx.to_onnx({'X': idi.astype(np.float32)})
        onnx2 = model_def.SerializeToString()
        self.assertEqual(onx.outputs, ['Y'])
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
        onx = OnnxReshape(
                OnnxReshape('X', np.array([1, -1], dtype=np.int64),
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


if __name__ == "__main__":
    unittest.main()
