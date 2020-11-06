# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from ..proto import onnx_proto
from ..common._apply_operation import (
    apply_slice, apply_cast, apply_identity, apply_reshape)
from ..common._registration import register_converter


def _common_convert_sklearn_zipmap(scope, operator, container):
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    to_type = onnx_proto.TensorProto.INT64

    if hasattr(operator, 'classlabels_int64s'):
        zipmap_attrs['classlabels_int64s'] = operator.classlabels_int64s
    elif hasattr(operator, 'classlabels_strings'):
        zipmap_attrs['classlabels_strings'] = operator.classlabels_strings
        to_type = onnx_proto.TensorProto.STRING

    if to_type == onnx_proto.TensorProto.STRING:
        apply_identity(scope, operator.inputs[0].full_name,
                       operator.outputs[0].full_name, container)
    else:
        apply_cast(scope, operator.inputs[0].full_name,
                   operator.outputs[0].full_name, container, to=to_type)
    return zipmap_attrs


def convert_sklearn_zipmap(scope, operator, container):
    zipmap_attrs = _common_convert_sklearn_zipmap(scope, operator, container)
    container.add_node('ZipMap', operator.inputs[1].full_name,
                       operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


def convert_sklearn_zipmap_columns(scope, operator, container):
    _common_convert_sklearn_zipmap(scope, operator, container)
    probs = operator.inputs[1].full_name
    for i in range(1, len(operator.outputs)):
        out = operator.outputs[i].full_name
        flat = scope.get_unique_variable_name(out)
        apply_slice(
            scope, probs, flat, container, starts=[i-1], ends=[i], axes=[1],
            operator_name=scope.get_unique_operator_name('Slice'))
        apply_reshape(
            scope, flat, out, container, desired_shape=(-1, ),
            operator_name=scope.get_unique_operator_name('reshape'))


register_converter('SklearnZipMap', convert_sklearn_zipmap)
register_converter('SklearnZipMapColumns', convert_sklearn_zipmap_columns)
