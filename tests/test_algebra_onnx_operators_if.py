# SPDX-License-Identifier: Apache-2.0

import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
import onnx
import onnx.helper
from onnx import TensorProto
from onnxruntime import InferenceSession, __version__ as ort_version
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import (
    OnnxAdd, OnnxSub, OnnxIf, OnnxGreater,
    OnnxReduceSum, OnnxMul, OnnxReduceMin)
from test_utils import TARGET_OPSET, TARGET_IR


ort_version = ".".join(ort_version.split('.')[:2])


class TestOnnxOperatorsIf(unittest.TestCase):

    @ignore_warnings(category=DeprecationWarning)
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.5.0'),
                     reason="too old onnxruntime")
    def test_onnx_if_test1(self):

        then_out = onnx.helper.make_tensor_value_info(
            'then_out', onnx.TensorProto.FLOAT, [5])
        else_out = onnx.helper.make_tensor_value_info(
            'else_out', onnx.TensorProto.FLOAT, [5])

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

        then_const_node = onnx.helper.make_node(
            'Constant', inputs=[], outputs=['then_out'],
            value=onnx.numpy_helper.from_array(x))

        else_const_node = onnx.helper.make_node(
            'Constant', inputs=[], outputs=['else_out'],
            value=onnx.numpy_helper.from_array(y))

        then_body = onnx.helper.make_graph(
            [then_const_node], 'then_body', [], [then_out])

        else_body = onnx.helper.make_graph(
            [else_const_node], 'else_body', [], [else_out])

        if_node = onnx.helper.make_node(
            'If', inputs=['cond'], outputs=['Z'],
            then_branch=then_body, else_branch=else_body)

        cond = onnx.helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
        Z = onnx.helper.make_tensor_value_info('Z', TensorProto.FLOAT, [None])
        graph_def = onnx.helper.make_graph([if_node], 'example', [cond], [Z])
        model_def = onnx.helper.make_model(graph_def, producer_name='skl2onnx')
        del model_def.opset_import[:]
        op_set = model_def.opset_import.add()
        op_set.domain = ''
        op_set.version = TARGET_OPSET
        model_def.ir_version = TARGET_IR

        cond = np.array(1).astype(bool)
        expected = x if cond else y
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'cond': cond})
        assert_almost_equal(expected, res[0])

    @ignore_warnings(category=DeprecationWarning)
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion('1.5.0'),
                     reason="too old onnxruntime")
    def test_onnx_if_test2(self):

        then_out = onnx.helper.make_tensor_value_info(
            'then_out', onnx.TensorProto.FLOAT, [5])
        else_out = onnx.helper.make_tensor_value_info(
            'else_out', onnx.TensorProto.FLOAT, [5])

        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

        then_const_node = onnx.helper.make_node(
            'Constant', inputs=[], outputs=['then_out'],
            value=onnx.numpy_helper.from_array(x))

        else_const_node = onnx.helper.make_node(
            'Identity', inputs=['Y'], outputs=['else_out'])

        then_body = onnx.helper.make_graph(
            [then_const_node], 'then_body', [], [then_out])

        else_body = onnx.helper.make_graph(
            [else_const_node], 'else_body', [], [else_out])

        if_node = onnx.helper.make_node(
            'If', inputs=['cond'], outputs=['Z'],
            then_branch=then_body, else_branch=else_body)

        cond = onnx.helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
        Y = onnx.helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        Z = onnx.helper.make_tensor_value_info('Z', TensorProto.FLOAT, [None])
        graph_def = onnx.helper.make_graph(
            [if_node], 'example', [cond, Y], [Z])
        model_def = onnx.helper.make_model(graph_def, producer_name='skl2onnx')
        del model_def.opset_import[:]
        op_set = model_def.opset_import.add()
        op_set.domain = ''
        op_set.version = TARGET_OPSET
        model_def.ir_version = TARGET_IR

        cond = np.array(1).astype(bool)
        expected = x if cond else y
        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'cond': cond, 'Y': y})
        assert_almost_equal(expected, res[0])

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_if_algebra_direct(self):

        opv = TARGET_OPSET
        x1 = np.array([[0, 3], [7, 0]], dtype=np.float32)
        x2 = np.array([[1, 0], [2, 0]], dtype=np.float32)

        node = OnnxAdd(
            'x1', 'x2', output_names=['absxythen'], op_version=opv)
        then_body = node.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['y'])
        model_def = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'x1': x1, 'x2': x2})
        assert_almost_equal(x1 + x2, res[0])

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_if_algebra_indirect(self):

        opv = TARGET_OPSET
        x1 = np.array([[0, 3], [7, 0]], dtype=np.float32)
        x2 = np.array([[1, 0], [2, 0]], dtype=np.float32)

        node_xy = OnnxMul(
            'x1', 'x2', op_version=opv, output_names=['xy'])
        node_then = OnnxAdd(
            'x1', 'xy', output_names=['absxythen'], op_version=opv)
        then_body = node_then.to_onnx(
            {'x1': x1, 'xy': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node_else = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node_else.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['y'],
                        global_context={'xy': node_xy})
        model_def = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'x1': x1, 'x2': x2})
        assert_almost_equal(x1 + x1 * x2, res[0])

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_if_algebra_indirect_unnamed(self):

        opv = TARGET_OPSET
        x1 = np.array([[0, 3], [7, 0]], dtype=np.float32)
        x2 = np.array([[1, 0], [2, 0]], dtype=np.float32)

        node_xy = OnnxMul('x1', 'x2', op_version=opv)
        node_then = OnnxAdd(
            'x1', 'xy', output_names=['absxythen'], op_version=opv)
        then_body = node_then.to_onnx(
            {'x1': x1, 'xy': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node_else = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node_else.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])
        del else_body.graph.input[:]
        del then_body.graph.input[:]

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['y'],
                        global_context={'xy': node_xy})
        model_def = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'x1': x1, 'x2': x2})
        assert_almost_equal(x1 + x1 * x2, res[0])

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_if_algebra_indirect_unnamed_clear_input(self):

        opv = TARGET_OPSET
        x1 = np.array([[0, 3], [7, 0]], dtype=np.float32)
        x2 = np.array([[1, 0], [2, 0]], dtype=np.float32)

        node_xy = OnnxMul('x1', 'x2', op_version=opv)
        node_then = OnnxAdd(
            'x1', 'xy', output_names=['absxythen'], op_version=opv)
        then_body = node_then.to_onnx(
            {'x1': x1, 'xy': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node_else = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node_else.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['y'],
                        global_context={'xy': node_xy},
                        clear_subgraph_inputs=True)
        model_def = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'x1': x1, 'x2': x2})
        assert_almost_equal(x1 + x1 * x2, res[0])

    @ignore_warnings(category=DeprecationWarning)
    def test_onnx_if_algebra_indirect_unnamed_clear_input_recursive(self):

        opv = TARGET_OPSET
        x1 = np.array([[0, 3], [7, 0]], dtype=np.float32)
        x2 = np.array([[1, 0], [2, 0]], dtype=np.float32)

        node_xy = OnnxMul('x1', 'x2', op_version=opv)
        node_then = OnnxAdd(
            'x1', 'xy', output_names=['absxythen'], op_version=opv)
        then_body = node_then.to_onnx(
            {'x1': x1, 'xy': x2}, target_opset=opv,
            outputs=[('absxythen', FloatTensorType())])
        node_else = OnnxSub(
            'x1', 'x2', output_names=['absxyelse'], op_version=opv)
        else_body = node_else.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('absxyelse', FloatTensorType())])

        cond = OnnxGreater(
            OnnxReduceSum('x1', op_version=opv),
            OnnxReduceSum('x2', op_version=opv),
            op_version=opv)
        ifnode = OnnxIf(cond, then_branch=then_body.graph,
                        else_branch=else_body.graph,
                        op_version=opv, output_names=['yt'],
                        clear_subgraph_inputs=True)
        subgraph = ifnode.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('yt', FloatTensorType())])

        cond2 = OnnxGreater(
            OnnxReduceMin('x1', op_version=opv),
            OnnxReduceMin('x2', op_version=opv),
            op_version=opv)
        ifnode2 = OnnxIf(cond2, then_branch=then_body.graph,
                         else_branch=subgraph.graph,
                         op_version=opv, output_names=['y'],
                         global_context={'xy': node_xy},
                         clear_subgraph_inputs=True)
        model_def = ifnode2.to_onnx(
            {'x1': x1, 'x2': x2}, target_opset=opv,
            outputs=[('y', FloatTensorType())])

        sess = InferenceSession(model_def.SerializeToString())
        res = sess.run(None, {'x1': x1, 'x2': x2})
        assert_almost_equal(x1 + x1 * x2, res[0])


if __name__ == "__main__":
    unittest.main()
