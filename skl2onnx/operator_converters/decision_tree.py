# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
import numpy as np
import six
from ..common._apply_operation import apply_cast
from ..common.data_types import Int64TensorType
from ..common._registration import register_converter
from ..common.tree_ensemble import add_tree_to_attribute_pairs
from ..common.tree_ensemble import get_default_tree_classifier_attribute_pairs
from ..common.tree_ensemble import get_default_tree_regressor_attribute_pairs
from ..proto import onnx_proto


def convert_sklearn_decision_tree_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)

    classes = op.classes_
    if all(isinstance(i, np.ndarray) for i in classes):
        classes = np.concatenate(classes)
    if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        attrs['classlabels_int64s'] = class_labels
    elif all(isinstance(i, (six.string_types, six.text_type))
             for i in classes):
        class_labels = [str(i) for i in classes]
        attrs['classlabels_strings'] = class_labels
    else:
        raise ValueError('Labels must be all integers or all strings.')

    add_tree_to_attribute_pairs(attrs, True, op.tree_, 0, 1., 0, True,
                                True, dtype=container.dtype)

    container.add_node(
        op_type, operator.input_full_names,
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain='ai.onnx.ml', **attrs)


def convert_sklearn_decision_tree_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = int(op.n_outputs_)
    add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False,
                                True, dtype=container.dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) == Int64TensorType:
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = [cast_input_name]

    container.add_node(op_type, input_name,
                       operator.output_full_names, op_domain='ai.onnx.ml',
                       **attrs)


register_converter('SklearnDecisionTreeClassifier',
                   convert_sklearn_decision_tree_classifier)
register_converter('SklearnDecisionTreeRegressor',
                   convert_sklearn_decision_tree_regressor)
register_converter('SklearnExtraTreeClassifier',
                   convert_sklearn_decision_tree_classifier)
register_converter('SklearnExtraTreeRegressor',
                   convert_sklearn_decision_tree_regressor)
