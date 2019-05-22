# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx_proto
from ..common._apply_operation import apply_cast, apply_identity
from ..common._registration import register_converter


def convert_sklearn_zipmap(scope, operator, container):
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
    container.add_node('ZipMap', operator.inputs[1].full_name,
                       operator.outputs[1].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


register_converter('SklearnZipMap', convert_sklearn_zipmap)
