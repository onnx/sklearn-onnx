# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_converter


def convert_sklearn_zipmap(scope, operator, container):
    zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
    if hasattr(operator, 'classlabels_int64s'):
        zipmap_attrs['classlabels_int64s'] = operator.classlabels_int64s
    elif hasttr(operator, 'classlabels_strings'):
        zipmap_attrs['classlabels_strings'] = operator.classlabels_strings
    container.add_node('ZipMap', operator.inputs[-1].full_name, operator.outputs[0].full_name,
                       op_domain='ai.onnx.ml', **zipmap_attrs)


register_converter('SklearnZipMap', convert_sklearn_zipmap)
