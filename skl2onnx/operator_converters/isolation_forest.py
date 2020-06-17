# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from ..common._apply_operation import (
    apply_cast,
    apply_concat,
    apply_mul,
    apply_reshape,
    apply_transpose,
)
from ..common._registration import register_converter
from ..common.data_types import BooleanTensorType, Int64TensorType
from ..common.tree_ensemble import (
    add_tree_to_attribute_pairs,
    get_default_tree_classifier_attribute_pairs,
    get_default_tree_regressor_attribute_pairs,
)
from ..proto import onnx_proto


def convert_sklearn_isolation_forest(
        scope, operator, container, op_type='TreeEnsembleRegressor',
        op_domain='ai.onnx.ml', op_version=1):
    op = operator.raw_operator

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1
    add_tree_to_attribute_pairs(attrs, False, op.tree_, 0, 1., 0, False,
                                True, dtype=container.dtype)

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name('cast_input')

        apply_cast(scope, operator.input_full_names, cast_input_name,
                   container, to=onnx_proto.TensorProto.FLOAT)
        input_name = [cast_input_name]

    container.add_node(
        op_type, input_name, operator.output_full_names,
        op_domain=op_domain, op_version=op_version, **attrs)


register_converter('SklearnIsolationForest',
                   convert_sklearn_isolation_forest)
