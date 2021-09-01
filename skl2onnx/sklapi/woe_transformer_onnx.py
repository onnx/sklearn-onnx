# SPDX-License-Identifier: Apache-2.0

from typing import List
import numpy as np
from onnx import TensorProto
from sklearn.base import BaseEstimator
from ..common.data_types import (
    Int64TensorType, FloatTensorType, DoubleTensorType)
from ..common._topology import Scope, Operator, Variable
from ..common._container import ModelComponentContainer
from ..common.utils import (
    check_input_and_output_types,
    check_input_and_output_numbers)
from .. import update_registered_converter
from .._supported_operators import _get_sklearn_operator_name
from ..algebra.onnx_ops import (
    OnnxIdentity, OnnxMatMul, OnnxGather, OnnxConcat, OnnxReshape,
    OnnxTreeEnsembleRegressor, OnnxOneHotEncoder, OnnxCast)
from .woe_transformer import WOETransformer


def woe_parser(scope: Scope, model: BaseEstimator,
               inputs: List[Variable], custom_parsers: dict = None):
    "ONNX parser for WOETransformer: defines the output type."
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    output = scope.declare_local_variable(
        "encoding", inputs[0].type.__class__())
    this_operator.inputs = inputs
    this_operator.outputs.append(output)
    return this_operator.outputs


def woe_shape_calculator(operator: Operator):
    "ONNX shape calculator for WOETransformer: defines the output shape."
    type_list = [Int64TensorType, FloatTensorType, DoubleTensorType]
    check_input_and_output_types(
        operator, good_input_types=type_list, good_output_types=type_list)
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=1)
    op = operator.raw_operator
    x = operator.inputs[0]
    N = x.get_first_dimension()
    C = 0
    for ext in op.intervals_:
        if ext is None:
            C += 1
        else:
            C += len(ext)
    operator.outputs[0].type.shape = [N, C]


class Tree:

    class Node:

        def __init__(self, parent, is_left, is_leaf, feature,
                     threshold, value):
            self.parent = parent
            self.is_left = is_left
            self.is_leaf = is_leaf
            self.feature = feature
            self.threshold = threshold
            self.value = value

        def __str__(self):
            return "Node(%s, %r, %r, %r, %r, %r)%s" % (
                self.parent if isinstance(self.parent, int)
                else "id%r" % id(self.parent),
                self.is_left, self.is_leaf, self.feature,
                self.threshold, self.value,
                "  # %s %r -> %r%s%s%s" % (
                    self.onnx_mode, self.onnx_threshold, self.onnx_value,
                    " -- %r" % self.intervals_
                    if hasattr(self, 'intervals_') else '',
                    " LL %r" % self.intervals_left_
                    if hasattr(self, 'intervals_left_') else '',
                    " RR %r" % self.intervals_right_
                    if hasattr(self, 'intervals_right_') else ''))

        @property
        def onnx_value(self):
            return self.value if self.value is not None else 0

        @property
        def onnx_threshold(self):
            if self.is_leaf:
                return self.threshold
            return self.threshold[0]

        @property
        def onnx_mode(self):
            # 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT',
            # 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
            if self.is_leaf:
                return 'LEAF'
            if self.threshold[1]:
                return 'BRANCH_LEQ'
            return 'BRANCH_LT'

        def is_on_left_side(self, x, leq):
            th = self.threshold[0]
            kind = self.onnx_mode
            if kind not in ('BRANCH_LEQ', 'BRANCH_LT'):
                raise NotImplementedError(
                    "Not implemented for mode %r." % kind)
            if x < th:
                return False
            if x > th:
                return True
            if kind == 'BRANCH_LEQ' and leq:
                return False
            if kind == 'BRANCH_LT' and not leq:
                return False
            return True

    def __init__(self):
        self.nodes = []

    def __str__(self):
        mapping = {}
        for n in self.nodes:
            mapping[str(id(n))] = str(len(mapping))
        rows = []
        for n in self.nodes:
            rows.append("id%r = %s" % (id(n), n))
        res = "\n".join(rows)
        for k, v in mapping.items():
            res = res.replace("id" + k, "n" + v)
        return res

    def add_node(self, parent, is_left, is_leaf, feature, threshold,
                 value=None):
        if is_leaf and value is None:
            raise ValueError("value must be specified when is_leaf=True.")
        if not is_leaf and value is not None:
            raise ValueError("value must not be specified when is_leaf=False.")
        node = Tree.Node(parent, is_left, is_leaf, feature, threshold, value)
        self.nodes.append(node)
        return node

    def update_feature(self, feature):
        "Change the feature index."
        for node in self.nodes:
            node.feature = feature

    def onnx_attributes(self):
        """
        See `TreeEnsembleRegressor
        <https://github.com/onnx/onnx/blob/master/docs/
        Operators-ml.md#ai.onnx.ml.TreeEnsembleRegressor>`_.
        """
        atts = dict(
            aggregate_function='SUM',
            base_values=[float(0)],
            n_targets=1,
            nodes_featureids=[n.feature for n in self.nodes],
            nodes_missing_value_tracks_true=[0 for n in self.nodes],
            nodes_modes=[n.onnx_mode for n in self.nodes],
            nodes_nodeids=[i for i in range(len(self.nodes))],
            nodes_treeids=[0 for n in self.nodes],
            nodes_values=[float(n.onnx_threshold) for n in self.nodes],
            post_transform='NONE',
            target_ids=[0 for n in self.nodes if n.onnx_mode == 'LEAF'],
            target_nodeids=[i for i, n in enumerate(self.nodes)
                            if n.onnx_mode == 'LEAF'],
            target_treeids=[0 for n in self.nodes
                            if n.onnx_mode == 'LEAF'],
            target_weights=[float(n.onnx_value) for n in self.nodes
                            if n.onnx_mode == 'LEAF'])

        ids = {id(n): (i, n) for i, n in enumerate(self.nodes)}
        nodes_truenodeids = [0 for n in self.nodes]  # right
        nodes_falsenodeids = [0 for n in self.nodes]  # left
        for i, n in enumerate(self.nodes):
            if n.parent == -1:
                continue
            idp = id(n.parent)
            val = ids[idp]
            if n.is_left:
                nodes_truenodeids[val[0]] = i
            else:
                nodes_falsenodeids[val[0]] = i
        atts.update(dict(
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_truenodeids=nodes_truenodeids))
        if len(atts['target_weights']) != len(set(atts['target_weights'])):
            raise RuntimeError(
                "All targets should be unique %r." % atts['target_weights'])
        return atts

    def mapping(self, intervals):
        """
        Maps every leaf target to the list of intervals
        it intersects. It creates attributes `intervals_`, `intervals_left_`,
        `intervals_rights_` as dictionary `{idx: interval}`
        each side intersects.
        """
        def process(node, intervals):
            if hasattr(node, 'intervals_'):
                return 0

            if node.parent is None or node.parent == -1:
                node.intervals_ = intervals
            else:
                if not hasattr(node.parent, 'intervals_'):
                    return 0
                node.intervals_ = (
                    node.parent.intervals_left_ if node.is_left
                    else node.parent.intervals_right_)

            if node.value is not None:
                # leaf
                return 1

            left = {}
            right = {}
            for k, v in node.intervals_.items():
                deca = node.is_on_left_side(v[0], v[2])
                decb = node.is_on_left_side(v[1], v[3])
                if not decb:
                    assert not deca
                    left[k] = v
                elif deca:
                    assert decb
                    right[k] = v
                elif decb and not deca:
                    left[k] = v
                    right[k] = v

            node.intervals_left_ = left
            node.intervals_right_ = right
            return 1

        for node in self.nodes:
            for at in ['intervals_', 'intervals_left_', 'intervals_right_']:
                if hasattr(node, at):
                    delattr(node, at)

        d_intervals = {i: t for i, t in enumerate(intervals)}
        changes = 1
        while changes > 0:
            changes = 0
            for node in self.nodes:
                changes += process(node, d_intervals)

        # final
        mapping = {}
        for node in self.nodes:
            if node.value is None:
                # not a leaf
                continue
            mapping[node.onnx_value] = node.intervals_
        return mapping


def digitize2tree(bins, right=False):
    ascending = len(bins) <= 1 or bins[0] < bins[1]

    if not ascending:
        bins2 = bins[::-1]
        cl = digitize2tree(bins2, right=right)
        n = len(bins)
        for i in range(cl.tree_.value.shape[0]):
            cl.tree_.value[i, 0, 0] = n - cl.tree_.value[i, 0, 0]
        return cl

    tree = Tree()
    values = []
    UNUSED = np.nan
    n_nodes = []

    def add_root(index):
        if index < 0 or index >= len(bins):
            raise IndexError(  # pragma: no cover
                "Unexpected index %d / len(bins)=%d." % (
                    index, len(bins)))
        parent = -1
        is_left = False
        is_leaf = False
        threshold = bins[index]
        n = tree.add_node(parent, is_left, is_leaf, 0, threshold)
        values.append(UNUSED)
        n_nodes.append(n)
        return n

    def add_nodes(parent, i, j, is_left):
        # add for bins[i:j] (j excluded)
        if is_left:
            # it means j is the parent split
            if i == j:
                # leaf
                n = tree.add_node(parent, is_left, True, 0, 0, value=i)
                n_nodes.append(n)
                values.append(i)
                return n
            if i + 1 == j:
                # split
                values.append(UNUSED)
                th = bins[i]
                n = tree.add_node(parent, is_left, False, 0, th)
                n_nodes.append(n)
                add_nodes(n, i, i, True)
                add_nodes(n, i, j, False)
                return n
            if i + 1 < j:
                # split
                values.append(UNUSED)
                index = (i + j) // 2
                th = bins[index]
                n = tree.add_node(parent, is_left, False, 0, th)
                n_nodes.append(n)
                add_nodes(n, i, index, True)
                add_nodes(n, index, j, False)
                return n
        else:
            # it means i is the parent split
            if i + 1 == j:
                # leaf
                values.append(j)
                n = tree.add_node(parent, is_left, True, 0, 0, value=j+1)
                n_nodes.append(n)
                return n
            if i + 1 < j:
                # split
                values.append(UNUSED)
                index = (i + j) // 2
                th = bins[index]
                n = tree.add_node(parent, is_left, False, 0, th)
                n_nodes.append(n)
                add_nodes(n, i, index, True)
                add_nodes(n, index, j, False)
                return n
        raise NotImplementedError(  # pragma: no cover
            "Unexpected case where i=%r, j=%r, is_left=%r." % (
                i, j, is_left))

    index = len(bins) // 2
    root = add_root(index)
    add_nodes(root, 0, index, True)
    add_nodes(root, index, len(bins), False)
    return tree


def woe_converter(scope: Scope, operator: Operator,
                  container: ModelComponentContainer):
    "ONNX Converter for WOETransformer."
    def mapping2matrix(mapping, value_mapping):
        rev = {v: k for k, v in enumerate(value_mapping)}
        rows = int(max(rev[k] for k in mapping)) + 1
        cols = max(max(v) if v else 0 for v in mapping.values()) + 1
        mat = np.zeros((rows, cols), dtype=np.int64)
        for k, intervals in mapping.items():
            for idx in intervals:
                mat[rev[k], idx] = 1

        # Remove all empty columns.
        total = mat.sum(axis=0, keepdims=0)
        return mat[:, total > 0].copy()

    op = operator.raw_operator
    X = operator.inputs[0]
    output = operator.outputs[0]
    opv = container.target_opset
    new_shape = np.array([-1, 1], dtype=np.int64)
    vector_shape = np.array([-1], dtype=np.int64)

    columns = []

    thresholds = op._decision_thresholds()
    conc = None
    for i, threshold in enumerate(thresholds):
        if threshold is None:
            # Passthrough columns
            index = np.array([i], dtype=np.int64)
            columns.append(
                OnnxReshape(
                    OnnxGather(X, index, op_version=opv, axis=1),
                    new_shape, op_version=opv))
            continue

        # encoding columns
        tree = digitize2tree(threshold)
        tree.update_feature(i)
        cats = list(sorted(set(int(n.onnx_value)
                               for n in tree.nodes if n.is_leaf)))
        mapping = tree.mapping(op.intervals_[i])
        mat_mapping = mapping2matrix(mapping, cats)
        # print("***********", i, threshold)
        # print(tree)
        # import pprint
        # pprint.pprint(mapping)
        # print("cats", cats)
        # print(mat_mapping)

        atts = tree.onnx_attributes()
        node = OnnxTreeEnsembleRegressor(
            X, op_version=1, domain='ai.onnx.ml', **atts)
        ohe = OnnxOneHotEncoder(
            OnnxReshape(node, vector_shape, op_version=opv),
            op_version=opv, cats_int64s=cats)
        ren = OnnxMatMul(
            OnnxCast(ohe, op_version=opv, to=TensorProto.INT64),
            mat_mapping, op_version=opv)
        cast = OnnxCast(ren, op_version=opv, to=TensorProto.FLOAT)
        columns.append(cast)

    conc = OnnxConcat(*columns, op_version=opv, axis=1)
    final = OnnxIdentity(conc, output_names=[output], op_version=opv)
    final.add_to(scope, container)


def register():
    "Register converter for WOETransformer."
    update_registered_converter(
        WOETransformer, "Skl2onnxWOETransformer",
        woe_shape_calculator, woe_converter, parser=woe_parser)
