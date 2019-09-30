# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from logging import getLogger
import copy
import numpy
from pandas import DataFrame
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from skl2onnx.third_party_skl.operator_converters.conv_lightgbm import (
    modify_tree_for_rule_in_set
)
from skl2onnx.third_party_skl.parsers.parse_lightgbm import (
    MockWrappedLightGbmBoosterClassifier
)
from skl2onnx.third_party_skl import register_converters
from skl2onnx import to_onnx
from onnxruntime import InferenceSession


def count_nodes(tree, done=None):
    if done is None:
        done = {}
    tid = id(tree)
    if tid in done:
        return 0
    done[tid] = tree
    nb = 1
    if 'right_child' in tree:
        nb += count_nodes(tree['right_child'], done)
    if 'left_child' in tree:
        nb += count_nodes(tree['left_child'], done)
    return nb


tree2_t1 = {
    'decision_type': '==',
    'default_left': True,
    'internal_count': 1612,
    'internal_value': 0,
    'left_child': {
        'decision_type': '<=',
        'default_left': True,
        'internal_count': 1367,
        'internal_value': 1.02414,
        'left_child': {
            'decision_type': '<=',
            'default_left': True,
            'internal_count': 623,
            'internal_value': 1.02414,
            'left_child': {
                'leaf_count': 253,
                'leaf_index': 0,
                'leaf_value': 3.7749963852295396},
            'missing_type': 'None',
            'right_child': {
                'leaf_count': 370,
                'leaf_index': 5,
                'leaf_value': 3.7749963852295396},
            'split_feature': 3,
            'split_gain': 1.7763600157738027e-15,
            'split_index': 4,
            'threshold': 3.5000000000000004},
        'missing_type': 'None',
        'right_child': {
            'decision_type': '<=',
            'default_left': True,
            'internal_count': 744,
            'internal_value': 1.02414,
            'left_child': {
                'leaf_count': 291,
                'leaf_index': 3,
                'leaf_value': 3.7749963852295396},
            'missing_type': 'None',
            'right_child': {
                'leaf_count': 453,
                'leaf_index': 4,
                'leaf_value': 3.7749963852295396},
            'split_feature': 3,
            'split_gain': 3.552710078910475e-15,
            'split_index': 3,
            'threshold': 3.5000000000000004},
        'split_feature': 2,
        'split_gain': 7.105429898699844e-15,
        'split_index': 2,
        'threshold': 3.5000000000000004},
    'missing_type': 'None',
    'right_child': {
        'decision_type': '<=',
        'default_left': True,
        'internal_count': 245,
        'internal_value': -5.7143,
        'left_child': {
            'leaf_count': 128,
            'leaf_index': 1,
            'leaf_value': 3.130106784685405},
        'missing_type': 'None',
        'right_child': {
            'leaf_count': 117,
            'leaf_index': 2,
            'leaf_value': 3.7749963852295396},
        'split_feature': 3,
        'split_gain': 234.05499267578125,
        'split_index': 1,
        'threshold': 6.500000000000001},
    'split_feature': 2,
    'split_gain': 217.14300537109375,
    'split_index': 0,
    'threshold': '8||9||10'}

tree2_t2 = {
    'decision_type': '<=',
    'default_left': True,
    'internal_count': 1612,
    'internal_value': 0,
    'left_child': {
        'leaf_count': 1367,
        'leaf_index': 0,
        'leaf_value': 0.05114685710677944},
    'missing_type': 'None',
    'right_child': {
        'decision_type': '<=',
        'default_left': True,
        'internal_count': 245,
        'internal_value': -3.89759,
        'left_child': {
             'leaf_count': 128,
             'leaf_index': 1,
             'leaf_value': -0.3177225912983217},
        'missing_type': 'None',
        'right_child': {
             'leaf_count': 117,
             'leaf_index': 2,
             'leaf_value': 0.05114685710677942},
        'split_feature': 3,
        'split_gain': 93.09839630126953,
        'split_index': 1,
        'threshold': 6.500000000000001},
    'split_feature': 2,
    'split_gain': 148.33299255371094,
    'split_index': 0,
    'threshold': 8.500000000000002}

tree2 = {'average_output': False,
         'feature_names': ['c1', 'c2', 'c3', 'c4'],
         'label_index': 0,
         'max_feature_idx': 3,
         'name': 'tree',
         'num_class': 1,
         'num_tree_per_iteration': 1,
         'objective': 'binary sigmoid:1',
         'pandas_categorical': None,
         'tree_info': [{'num_cat': 0,
                        'num_leaves': 6,
                        'shrinkage': 1,
                        'tree_index': 0,
                        'tree_structure': tree2_t1},
                       {'num_cat': 0,
                        'num_leaves': 3,
                        'shrinkage': 0.05,
                        'tree_index': 1,
                        'tree_structure': tree2_t2}],
         'version': 'v2'}


class TestLightGbmTreeStructur(unittest.TestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    def test_onnxrt_python_lightgbm_categorical(self):
        val = {'decision_type': '==',
               'default_left': True,
               'internal_count': 6805,
               'internal_value': 0.117558,
               'left_child': {'leaf_count': 4293,
                              'leaf_index': 18,
                              'leaf_value': 0.003519117642745049},
               'missing_type': 'None',
               'right_child': {'leaf_count': 2512,
                               'leaf_index': 25,
                               'leaf_value': 0.012305307958365394},
               'split_feature': 24,
               'split_gain': 12.233599662780762,
               'split_index': 24,
               'threshold': '10||12||13'}

        t2 = {'decision_type': '==',
              'default_left': True,
              'internal_count': 6805,
              'internal_value': 0.117558,
              'left_child': {'leaf_count': 4293,
                               'leaf_index': 18,
                               'leaf_value': 0.003519117642745049},
              'missing_type': 'None',
              'right_child': {'decision_type': '==',
                              'default_left': True,
                              'internal_count': 6805,
                              'internal_value': 0.117558,
                              'left_child': {
                                'leaf_count': 4293,
                                'leaf_index': 18,
                                'leaf_value': 0.003519117642745049},
                              'missing_type': 'None',
                              'right_child': {
                                'leaf_count': 2512,
                                'leaf_index': 25,
                                'leaf_value': 0.012305307958365394},
                              'split_feature': 24,
                              'split_gain': 12.233599662780762,
                              'split_index': 24,
                              'threshold': 13},
              'split_feature': 24,
              'split_gain': 12.233599662780762,
              'split_index': 24,
              'threshold': 12}

        exp = {'decision_type': '==',
               'default_left': True,
               'internal_count': 6805,
               'internal_value': 0.117558,
               'left_child': {'leaf_count': 4293,
                                'leaf_index': 18,
                                'leaf_value': 0.003519117642745049},
               'missing_type': 'None',
               'right_child': t2,
               'split_feature': 24,
               'split_gain': 12.233599662780762,
               'split_index': 24,
               'threshold': 10}

        nb1 = count_nodes(val)
        modify_tree_for_rule_in_set(val)
        nb2 = count_nodes(val)
        self.assertEqual(nb1, 3)
        self.assertEqual(nb2, 5)
        sval = str(val)
        self.assertNotIn('||', sval)
        self.maxDiff = None
        self.assertEqual(exp, val)

    def test_onnxrt_python_lightgbm_categorical2(self):
        val = copy.deepcopy(tree2)
        nb1 = sum(count_nodes(t['tree_structure']) for t in val['tree_info'])
        modify_tree_for_rule_in_set(val)
        nb2 = sum(count_nodes(t['tree_structure']) for t in val['tree_info'])
        self.assertEqual(nb1, 16)
        self.assertEqual(nb2, 18)

    def test_mock_lightgbm(self):

        tree = copy.deepcopy(tree2)
        nb1 = sum(count_nodes(t['tree_structure']) for t in tree['tree_info'])
        model = MockWrappedLightGbmBoosterClassifier(tree)
        nb2 = sum(count_nodes(t['tree_structure']) for t in tree['tree_info'])
        self.assertEqual(nb1, nb2)
        self.assertEqual(nb1, 16)
        onx = to_onnx(model, initial_types=[('x', FloatTensorType([None, 4]))])
        self.assertTrue(model.visited)

        for n in onx.graph.node:
            if n.op_type != 'TreeEnsembleClassifier':
                continue
            att = n.attribute
            for k in att:
                if k.name != 'nodes_modes':
                    continue
                values = k.strings
                nbnodes = len(values)
        self.assertEqual(nbnodes, 18)

        iris = load_iris()
        X = iris.data
        X = (X * 10).astype(numpy.int32).astype(numpy.float32)

        oif = InferenceSession(onx.SerializeToString())
        pred = oif.run(None, {'x': X})
        label = pred[0]
        self.assertEqual(label.shape, (X.shape[0], ))
        prob = DataFrame(pred[1]).values
        self.assertEqual(prob.shape, (X.shape[0], 2))


if __name__ == "__main__":
    unittest.main()
