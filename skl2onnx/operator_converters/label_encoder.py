# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._registration import register_converter


def convert_sklearn_label_encoder(scope, operator, container):
    op = operator.raw_operator
    op_type = 'LabelEncoder'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    classes = op.classes_
    if np.issubdtype(classes.dtype, np.floating):
        attrs['keys_floats'] = classes
    elif np.issubdtype(classes.dtype, np.signedinteger):
        attrs['keys_int64s'] = classes
    else:
        attrs['keys_strings'] = np.array([s.encode('utf-8') for s in classes])
    attrs['values_int64s'] = np.arange(len(classes))

    cop = container.target_opset_any_domain('ai.onnx.ml')
    if cop is not None and cop < 2:
        raise RuntimeError(
            "LabelEncoder requires at least opset 2 for domain 'ai.onnx.ml' "
            "not {}".format(cop))

    container.add_node(op_type, operator.input_full_names,
                       operator.output_full_names, op_domain='ai.onnx.ml',
                       op_version=2, **attrs)


register_converter('SklearnLabelEncoder', convert_sklearn_label_encoder)
